import pickle
import random as random_py
import time

import jax
import jax.numpy as jnp
import haiku as hk

import numpy as np
import optax
import seqtools
from jax import jit, random

try:
    from bayesian_benchmarks.data import get_regression_data
    from bayesian_benchmarks.database_utils import Database
except:
    print('WARNING: bayesian_benchmarks could not be loaded.')

from fsvi.utils import utils
from fsvi.utils.utils_training import Training
from fsvi.utils import utils_logging
from fsvi.utils import datasets
from fsvi.utils.utils import get_minibatch, initialize_random_keys

dtype_default = jnp.float32
# np.set_printoptions(suppress=True)
eps = 1e-6


def regression(
    prior_mean: str,
    prior_cov: str,
    batch_size: int,
    epochs: int,
    seed: int,
    save_path: str,
    save: bool,
    model_type: str,
    **kwargs,
):
    kh = initialize_random_keys(seed=seed)
    rng_key, rng_key_train, rng_key_test = random.split(kh.next_key(), 3)

    plot_progress = False

    # LOAD DATA
    # TODO: move to datasets (there's already a stub there)
    n_test = 500
    x_test_lim = 10
    if 'snelson' in kwargs['task']:
        '''
        --data_training snelson --model mlp_fsvi --architecture fc_100_100 --activation tanh --epochs 200000 --learning_rate 5e-4 --optimizer adam --batch_size 0 --prior_mean 0 --prior_cov 10 --prior_type bnn_induced_prior --inducing_points 10 --ind_lim ind_-8_8 --inducing_input_type uniform_train_rand_0.5 --prior_type fixed --kl_scale none --n_samples 5 --tau 0. --logging_frequency 100 --seed 0 --debug --save --save_path tmp
        stochastic_linearization = True ?
        stochastic_linearization_prior = True ?
        '''
        x, y, x_test, inducing_inputs_, noise_std = datasets.snelson(n_test=n_test, x_test_lim=x_test_lim, standardize_x=True, standardize_y=True)

        x_mean = x.mean()
        x_std = x.std()

        y_mean = y.mean()
        y_std = y.std()

        x_train = x
        y_train = y
        x_test_features = x_test
        y_test = None
    elif 'solar' in kwargs['task']:
        x, y, x_test, noise_std = datasets.load_solar(n_test=n_test)
        x_train = x
        y_train = y
        x_test_features = x_test
        y_test = None
    elif 'oat1d' in kwargs['task']:
        '''
        --data_training oat1d --model mlp_fsvi --architecture fc_100_100 --activation tanh --epochs 200000 --learning_rate 1e-3 --optimizer adam --batch_size 0 --prior_mean 0 --prior_cov 0.1 --prior_type bnn_induced --inducing_points 10 --ind_lim ind_-10_10 --inducing_input_type uniform_train_rand_0.5 --kl_scale none --n_samples 5 --tau 0. --logging_frequency 100 --seed 0 --debug --save --save_path tmp
        stochastic_linearization = True
        stochastic_linearization_prior = True
        '''
        x_train1 = np.linspace(-7, -4, 40)
        y_train1 = np.sin(x_train1 * np.pi * 0.5 - 2.5) * 4.7 - 1.2

        x_train2 = np.linspace(3,8,40)
        y_train2 = np.sin(x_train2 * np.pi * 0.58 - 0.5) * 1.6 - 2.7

        x = np.concatenate([x_train1, x_train2], 0)[:, None]
        x_mean = x.mean()
        x_std = x.std()
        x = (x - x_mean) / x_std
        x_train = x

        y = np.concatenate([y_train1, y_train2], 0)[:, None]
        y_mean = y.mean()
        y_std = y.std()
        y_train = (y - y_mean) / y_std

        x_test = np.linspace(-x_test_lim, x_test_lim, n_test)[:, None]
        x_test_features = x_test
        y_test = None

        noise_std = 0.001
    elif 'subspace_inference' in kwargs['task']:
        '''
        --data_training subspace_inference --model mlp_fsvi --architecture fc_100_100 --activation tanh --epochs 200000 --learning_rate 1e-3 --optimizer adam --batch_size 0 --prior_mean 0 --prior_cov 10 --prior_type fixed --inducing_points 10 --ind_lim ind_-10_10 --inducing_input_type uniform_train_rand_0.5 --kl_scale none --n_samples 5 --tau 0. --logging_frequency 100 --seed 0 --debug --save --save_path tmp
        stochastic_linearization = True
        stochastic_linearization_prior = NA
        '''
        def features(x):
            return np.hstack([x[:, None] / 2.0, (x[:, None] / 2.0) ** 2])

        data = np.load("data/subspace_inference.npy")
        x, y = data[:, 0], data[:, 1]
        y = y[:, None]

        x_mean = x.mean()
        x_std = x.std()
        x = (x - x_mean) / x_std
        x_train = features(x)

        y_mean = y.mean()
        y_std = y.std()
        y_train = (y - y_mean) / y_std

        x_test = np.linspace(-x_test_lim, x_test_lim, n_test)
        x_test_features = features(x_test)
        y_test = None

        noise_std = 0.01
    elif 'uci' in kwargs['task']:
        dataset = kwargs['task'].split("uci_", 1)[1].split("_", 1)[0]
        kwargs['data_training'] = 'uci'
        data = get_regression_data(dataset, split=seed)
        x_train, y_train, x_test, y_test = data.X_train, data.Y_train, data.X_test, data.Y_test
        x, y = x_train, y_train
        noise_std = kwargs['noise_std']
        x_test_features = x_test

        x_mean = data.X_mean
        x_std = data.X_std
        y_mean = data.Y_mean
        y_std = data.Y_std
    else:
        ValueError('No valid dataset specified.')

    #%%

    # TODO: remove hardcoding
    noise_var = noise_std ** 2

    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    input_shape = [1, input_dim]

    # DEFINE NUMPY TRAINLOADER  # TODO: test implementation
    n_train = x_train.shape[0]
    train_dataset = seqtools.collate([x_train, y_train])
    if batch_size == 0:
        batch_size = n_train
        _trainloader = seqtools.batch(train_dataset, batch_size, collate_fn=datasets.collate_fn)
    else:
        _trainloader = seqtools.batch(train_dataset, batch_size, collate_fn=datasets.collate_fn)

    trainloader = []
    for i, data in enumerate(_trainloader, 0):
        x_batch = np.array(data[0], dtype=dtype_default)
        y_batch = np.array(data[1], dtype=dtype_default)
        trainloader.append([x_batch, y_batch])

    permutation = np.random.permutation(x_train.shape[0])
    x_train_permuted = x_train[permutation,:]

    val_frac = 0.0
    kwargs['stochastic_linearization'] = False

    # INITIALIZE TRAINING CLASS
    training = Training(
        input_shape=input_shape,
        output_dim=output_dim,
        n_train=n_train,
        epochs=epochs,
        batch_size=batch_size,
        n_batches=n_train // batch_size,
        # TODO: unify the run.py for classification ood
        full_ntk=kwargs["full_cov"],
        model_type=model_type,
        **kwargs,
    )

    # INITIALIZE MODEL
    (model, init_fn, apply_fn, state, params) = training.initialize_model(
        rng_key=rng_key, x_train=x_train_permuted, x_ood=[x_train],
    )

    # INITIALIZE OPTIMIZATION
    (
        opt,
        opt_state,
        get_trainable_params,
        get_variational_and_model_params,
        metrics,
        loss,
        kl_evaluation,
        log_likelihood_evaluation,
        nll_grad_evaluation,
        task_evaluation,
        prediction_type,
    ) = training.initialize_optimization(
        model=model,
        apply_fn=apply_fn,
        params_init=params,
        state=state,
        rng_key=rng_key,
    )

    # INITIALIZE KL INPUT FUNCTIONS
    inducing_input_fn, prior_fn = training.kl_input_functions(
        apply_fn=apply_fn,
        predict_f=model.predict_f,
        predict_f_deterministic=model.predict_f_deterministic,
        state=state,
        params=params,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        rng_key=rng_key,
        x_ood=[x_train],
        y_ood=[x_train],
    )

    # INITIALIZE LOGGING CLASS
    epoch_start = 0
    logging = utils_logging.Logging(
        model=model,
        apply_fn=apply_fn,
        metrics=metrics,
        loss=loss,
        kl_evaluation=kl_evaluation,
        log_likelihood_evaluation=log_likelihood_evaluation,
        nll_grad_evaluation=nll_grad_evaluation,
        task_evaluation=task_evaluation,
        epoch_start=epoch_start,
        x_train_permuted=x_train,
        y_train_permuted=y_train,
        x_test=x_test,
        y_test=y_test,
        x_ood=[x_train],
        n_train=n_train,
        val_frac=val_frac,
        epochs=epochs,
        save=save,
        save_path=save_path,
        model_type=model_type,
        **kwargs,
    )

    # TODO: move elsewhere
    if 'uci' in kwargs['task']:
        training_progress = logging.print_regression
    else:
        training_progress = logging.plot_1d

    @jit
    def update(
        params,
        params_feature,
        state,
        opt_state_mu,
        opt_state_var,
        x_batch,
        y_batch,
        inducing_inputs,
        rng_key,
        prior_cov,
    ):
        trainable_params, non_trainable_params = get_trainable_params(params)
        variational_params, model_params = get_variational_and_model_params(params)
        _prior_mean, _prior_cov = prior_fn(
            inducing_inputs=inducing_inputs,
            model_params=model_params,
            rng_key=rng_key,
            prior_cov=prior_cov,
            state=state,
        )

        grads, new_state = jax.grad(loss, argnums=0, has_aux=True)(
            trainable_params,
            [non_trainable_params,params_feature],
            state,
            _prior_mean,
            _prior_cov,
            x_batch,
            y_batch,
            inducing_inputs,
            rng_key,
        )
        zero_grads = jax.tree_map(lambda x: x * 0., non_trainable_params)
        grads = jax.tree_map(lambda x: x * 1., grads)
        grads_full = hk.data_structures.merge(grads, zero_grads)

        if opt_var is None:
            updates, opt_state_mu = opt_mu.update(grads_full, opt_state_mu)
            new_params = optax.apply_updates(params, updates)
        else:
            grads_full_mu, grads_full_var, _ = partition_params(grads_full)
            updates_mu, opt_mu_state = opt_mu.update(grads_full_mu, opt_state_mu)
            updates_var, opt_var_state = opt_var.update(grads_full_var, opt_state_var)
            updates = hk.data_structures.merge(updates_mu, updates_var)
            new_params = optax.apply_updates(params, updates)

        if loss_tracking:
            trainable_params_new, non_trainable_params_new = get_trainable_params(new_params)

            loss_val_0, _ = loss(
                trainable_params,
                [non_trainable_params, params_feature],
                state,
                _prior_mean,
                _prior_cov,
                x_batch,
                y_batch,
                inducing_inputs,
                rng_key
            )

            loss_val_1, _ = loss(
                trainable_params_new,
                [non_trainable_params_new, params_feature],
                state,
                _prior_mean,
                _prior_cov,
                x_batch,
                y_batch,
                inducing_inputs,
                rng_key
            )

            loss_val_2, _ = loss(
                trainable_params_new,
                [non_trainable_params_new, new_params],
                state,
                _prior_mean,
                _prior_cov,
                x_batch,
                y_batch,
                inducing_inputs,
                rng_key
            )
        else:
            loss_val_0, loss_val_1, loss_val_2 = 0, 0, 0

        is_nan = jnp.isnan(jax.flatten_util.ravel_pytree(new_params)[0]).any()
        nan_check = [is_nan, params, new_params, grads_full, loss_val_0, loss_val_1, loss_val_2]

        params = new_params

        return params, opt_state_mu, opt_state_var, new_state, nan_check

    if not isinstance(opt, list):
        opt_mu, opt_var = opt, None
        opt_state_mu, opt_state_var = opt_state, None
    else:
        opt_mu, opt_var = opt
        opt_state_mu, opt_state_var = opt_state

    num = 1000
    grads_list = []
    params_list = []
    loss_0_list = []
    loss_1_list = []
    loss_2_list = []
    loss_diff_param_list = []
    loss_diff_feature_list = []
    loss_tracking = False

    print(f"\n--- Training for {epochs} epochs ---\n")
    for epoch in range(epochs):
        logging.t0 = time.time()

        for i, data in enumerate(trainloader, 0):
            if logging.feature_update or (i+1) % kwargs["feature_update"] == 0 or epoch < 0:
                params_feature = params
                logging.feature_update = False
            rng_key_train, _ = random.split(rng_key_train)

            x_batch, y_batch = get_minibatch(
                data, output_dim, input_shape, prediction_type
            )

            # permutation = jax.random.permutation(key=rng_key_train, x=x_batch.shape[0])
            # x_batch_test = x_test[permutation]
            # x_batch_inducing = jnp.concatenate([x_batch, x_batch_test], axis=0)
            inducing_inputs = inducing_input_fn(x_batch, rng_key_train)
            if 'mlp' in model_type:
                x_batch = x_batch.reshape(batch_size, -1)
                inducing_inputs = inducing_inputs.reshape(inducing_inputs.shape[0], -1)

            params, opt_state_mu, opt_state_var, state, nan_check = update(
                params,
                params_feature,
                state,
                opt_state_mu,
                opt_state_var,
                x_batch,
                y_batch,
                inducing_inputs,
                rng_key_train,
                dtype_default(prior_cov),
            )

            assert not nan_check[0]
            # params_list.append(np.array(jnp.abs(jax.flatten_util.ravel_pytree(nan_check[1])[0])).max())
            # grads_list.append(np.array(jnp.abs(jax.flatten_util.ravel_pytree(nan_check[3])[0])).max())
            # loss_0_list.append(nan_check[4])
            # loss_1_list.append(nan_check[5])
            # loss_2_list.append(nan_check[6])
            # loss_diff_param_list.append(nan_check[4] - nan_check[5])
            # loss_diff_feature_list.append(nan_check[5] - nan_check[6])

        training_progress(
            epoch = epoch,
            params = params,
            state = state,
            rng_key = rng_key,
            x = x,
            x_train = x_train,
            x_test_features = x_test_features,
            inducing_inputs = inducing_inputs,
            y_train = y_train,
            prior_mean = prior_mean,
            prior_cov = prior_cov,
            noise_var = noise_var,
            y_std = y_std,
            plot_progress = plot_progress,
            prior_fn = prior_fn,
        )

