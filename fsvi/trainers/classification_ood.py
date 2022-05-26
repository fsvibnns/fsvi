import os
from functools import partial
import time
import pickle
import random as random_py
from copy import copy

import numpy as np

from jax import jit, random
import jax.numpy as jnp
import haiku as hk

import optax

import wandb

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
path = dname + "/../.."
# print(f'Setting working directory to {path}\n')
os.chdir(path)

from fsvi.utils import utils_logging
from fsvi.utils import datasets
from fsvi.utils.utils import get_minibatch, initialize_random_keys
from fsvi.utils.utils_training import Training
from fsvi.utils.haiku_mod import partition_params

import jax
from jax import jit

from tensorflow_probability.substrates import jax as tfp

dtype_default = jnp.float32
tfd = tfp.distributions


def classification_ood(
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
    if kwargs['wandb_project'] != 'not_specified':
        wandb.config = copy(kwargs)
        wandb.init(
            project=kwargs['wandb_project'],
            entity="tgjr-research",
            config=kwargs
        )

    kh = initialize_random_keys(seed=seed)
    rng_key, rng_key_train, rng_key_test = random.split(kh.next_key(), 3)

    # LOAD DATA
    val_frac = 0.0
    (
        trainloader,
        x_train_permuted,
        y_train_permuted,
        x_test,
        y_test,
        x_ood,
        y_ood,
        x_inducing_inputs_ood_list,
        y_inducing_inputs_ood_list,
        input_shape,
        input_dim,
        output_dim,
        n_train,
        n_batches,
    ) = datasets.load_data(
        seed=seed,
        val_frac=val_frac,
        batch_size=batch_size,
        model_type=model_type,
        **kwargs,
    )

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
        rng_key=rng_key, x_train=x_train_permuted, x_ood=x_inducing_inputs_ood_list,
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
        x_ood=x_inducing_inputs_ood_list,
        y_ood=y_inducing_inputs_ood_list,
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
        x_train_permuted=x_train_permuted,
        y_train_permuted=y_train_permuted,
        x_test=x_test,
        y_test=y_test,
        x_ood=x_ood,
        n_train=n_train,
        val_frac=val_frac,
        epochs=epochs,
        save=save,
        save_path=save_path,
        model_type=model_type,
        **kwargs,
    )

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

            if epoch >= num:
                logging.T = time.time()
                logging.training_progress(i, params, params, state, rng_key_test, inducing_input_fn, prior_fn)

            assert not nan_check[0]
            # params_list.append(np.array(jnp.abs(jax.flatten_util.ravel_pytree(nan_check[1])[0])).max())
            # grads_list.append(np.array(jnp.abs(jax.flatten_util.ravel_pytree(nan_check[3])[0])).max())
            # loss_0_list.append(nan_check[4])
            # loss_1_list.append(nan_check[5])
            # loss_2_list.append(nan_check[6])
            # loss_diff_param_list.append(nan_check[4] - nan_check[5])
            # loss_diff_feature_list.append(nan_check[5] - nan_check[6])

        logging.T = time.time()
        logging.training_progress(epoch, params, params, state, rng_key_test, inducing_input_fn, prior_fn)

        # plot = False
        # if plot:
        #     import matplotlib.pyplot as plt
        #     plt.hist(params_list); plt.show()
        #     plt.hist(grads_list); plt.show()
        #     plt.hist(loss_0_list); plt.show()
        #     plt.hist(loss_1_list); plt.show()
        #     plt.hist(loss_2_list); plt.show()
        #     plt.hist(loss_diff_param_list); plt.show()
        #     plt.hist(loss_diff_feature_list); plt.show()

        if epoch + 1 == kwargs["start_var_opt"]:
            # TODO: For debugging, delete later:
            from jax import config
            config.update('jax_disable_jit', True)
            config.update('jax_disable_jit', False)

            training.start_var_opt = 0
            # REINITIALIZE OPTIMIZATION
            (
                _opt,
                _opt_state,
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
                prior_mean, prior_cov = prior_fn(
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
                    prior_mean,
                    prior_cov,
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

                params = new_params

                return params, opt_state_mu, opt_state_var, new_state

            with open(os.path.join(kwargs['run_folder'], "params_pickle_pre_reinit"), "wb") as file:
                to_save = {
                    "params": params,
                    "state": state,
                    "kwargs": kwargs,
                }
                pickle.dump(to_save, file)

            print(f"Reinitialization complete at epoch {epoch+1}.")

    # # TODO: Decide if we want to keep the following three lines of code
    # if save:
    #     with open(f"params_pickle_test_01", "wb") as file:
    #         pickle.dump(params, file)
    #     with open(f"state_pickle_test_01", "wb") as file:
    #         pickle.dump(state, file)
    #
    # # saving under the logging setup by `create_logdir`
    # with open(os.path.join(kwargs['run_folder'], "params_pickle"), "wb") as file:
    #     to_save = {
    #         "params": params,
    #         "state": state,
    #         "kwargs": kwargs,
    #     }
    #     pickle.dump(to_save, file)
