import os
import time
import pickle
import random as random_py

import numpy as np

from jax import jit, random
import jax.numpy as jnp
import haiku as hk

import optax

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
path = dname + "/../.."
# print(f'Setting working directory to {path}\n')
os.chdir(path)

from fsvi.utils import utils_logging
from fsvi.utils import datasets
from fsvi.utils.utils import get_minibatch, initialize_random_keys
from fsvi.utils.utils_training import Training

import jax
from jax import jit

from tensorflow_probability.substrates import jax as tfp

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
        batch_size=batch_size,
        n_batches=n_train // batch_size,
        # TODO: unify the run.py for classification ood
        full_ntk=False,
        model_type=model_type,
        **kwargs,
    )

    # INITIALIZE MODEL
    (model, init_fn, apply_fn, state, params) = training.initialize_model(
        rng_key=rng_key
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
        predict_f_deterministic=model.predict_f_deterministic,
        state=state,
        params=params,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        rng_key=rng_key,
        x_ood=x_ood,
    )

    # INITIALIZE LOGGING CLASS
    epoch_start = 0
    logging = utils_logging.Logging(
        model=model,
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

    def swap_layer(params: hk.Params, subset: hk.Params):
        mutable_params = hk.data_structures.to_mutable_dict(params)
        mutable_layer = hk.data_structures.to_mutable_dict(subset)
        mutable_params.update(mutable_layer)
        return hk.data_structures.to_immutable_dict(mutable_params)

    def swap_layer_v2(params: hk.Params, subset: hk.Params):
        mutable_params = hk.data_structures.to_mutable_dict(params)
        mutable_layer = hk.data_structures.to_mutable_dict(subset)
        custom_update(mutable_params, mutable_layer)
        return hk.data_structures.to_immutable_dict(mutable_params)

    def custom_update(params1, params2):
        """
        Update d1 with d2, only variable that exists in d2 will be updated, other variables
        in d1 remain the same.
        """
        for module, module_dict in params2.items():
            for variable, array in module_dict.items():
                params1[module][variable] = array

    @jit
    def update(
        params,
        state,
        opt_state,
        x_batch,
        y_batch,
        inducing_inputs,
        rng_key,
    ):
        trainable_params, non_trainable_params = get_trainable_params(params)
        variational_params, model_params = get_variational_and_model_params(params)
        prior_mean, prior_cov = prior_fn(
            inducing_inputs=inducing_inputs,
            model_params=model_params,
            rng_key=rng_key,
        )

        grads, new_state = jax.grad(loss, argnums=0, has_aux=True)(
            trainable_params,
            non_trainable_params,
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
        updates, opt_state = opt.update(grads_full, opt_state)
        new_params = optax.apply_updates(params, updates)
        params = new_params

        return params, opt_state, new_state

    p_reinitialize = 0.001
    j = 1
    m = 1000

    print(f"\n--- Training for {epochs} epochs ---\n")
    for epoch in range(epochs):
        logging.t0 = time.time()

        for i, data in enumerate(trainloader, 0):
            rng_key_train, _ = random.split(rng_key_train)

            x_batch, y_batch = get_minibatch(
                data, output_dim, input_shape, prediction_type
            )
            inducing_inputs = inducing_input_fn(x_batch, rng_key_train)
            if 'mlp' in model_type:
                x_batch = x_batch.reshape(batch_size, -1)
                inducing_inputs = inducing_inputs.reshape(inducing_inputs.shape[0], -1)

            params, opt_state, state = update(
                params,
                state,
                opt_state,
                x_batch,
                y_batch,
                inducing_inputs,
                rng_key_train,
            )

            reinitialize = np.random.binomial(1, p_reinitialize)
            j += 1

            if j % m == 0:
            # if False:
                logging.T = time.time()
                logging.training_progress(epoch, params, state, rng_key_test)
                new_params = init_fn(rng_key_train, x_batch, rng_key, model.stochastic_parameters, is_training=True)[0]

                init_layer = np.random.choice(list(params.keys()))
                # init_layer = list(params.keys())[-1]

                # if "conv" in init_layer:
                #     param_type = ['b_mu', 'w_mu']
                # else:
                #     param_type = ['b_mu', 'b_logvar', 'w_mu', 'w_logvar']
                param_type = ['b_mu', 'w_mu']
                # param_type = ['b_logvar', 'w_logvar']

                init_affine = np.random.choice(param_type)
                # init_affine = param_type

                predicate = lambda module_name, name, value: module_name == init_layer and name == init_affine
                subset_params = hk.data_structures.filter(predicate, new_params)
                partially_reinitialized_params = swap_layer_v2(params, subset=subset_params)

                print(f"Reinitializing layer {init_layer} + {init_affine}")
                params = partially_reinitialized_params

                # logging.training_progress(epoch, params, state, rng_key_test)
                print(f"Reinitialized layer {init_layer} + {init_affine}")

                logging.T = time.time()
                logging.training_progress(epoch, params, state, rng_key_test)


        logging.T = time.time()
        logging.training_progress(epoch, params, state, rng_key_test)
        # logging.training_progress_large(epoch, params, state, x_batch, y_batch, prior_mean, prior_cov, inducing_inputs, rng_key_test)

        # logging.log_training_progress(
        #     epoch, params, state, x_batch, y_batch, x_test, y_test, x_ood,
        #     prior_mean, prior_cov, inducing_inputs, rng_key_test
        # )
        #
        # logging.log_training_metrics(
        #     epoch, x_batch, y_batch, x_test, y_test, x_ood, y_ood, inducing_inputs,
        #     params, state, prior_mean, prior_cov, rng_key_test
        # )

    if save:
        with open(f"{save_path}/params_pickle", "wb") as file:
            pickle.dump(params, file)
