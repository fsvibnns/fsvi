import os
import time
import pickle
import random as random_py

import numpy as np

from jax import jit, random

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


def causal_inference(
    prior_mean: str,
    prior_cov: str,
    batch_size: int,
    epochs: int,
    seed: int,
    save_path: str,
    save: bool,
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
        **kwargs,
    )

    # INITIALIZE TRAINING CLASS
    training = Training(
        input_shape=input_shape,
        output_dim=output_dim,
        n_train=n_train,
        batch_size=batch_size,
        n_batches=n_train // batch_size,
        stochastic_linearization=True,
        full_ntk=False,
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
        **kwargs,
    )

    @jit
    def update(
        params,
        state,
        opt_state,
        prior_mean,
        prior_cov,
        x_batch,
        y_batch,
        inducing_inputs,
        rng_key,
    ):
        grads, new_state = jax.grad(loss, argnums=0)(
            params,
            state,
            prior_mean,
            prior_cov,
            x_batch,
            y_batch,
            inducing_inputs,
            rng_key,
        )
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, new_state

    print(f"\n--- Training for {epochs} epochs ---\n")
    for epoch in range(epochs):
        logging.t0 = time.time()

        for i, data in enumerate(trainloader, 0):
            rng_key_train, _ = random.split(rng_key_train)

            x_batch, y_batch = get_minibatch(
                data, output_dim, input_shape, prediction_type
            )
            inducing_inputs = inducing_input_fn(x_batch, rng_key_train)
            prior_mean, prior_cov = prior_fn(inducing_inputs=inducing_inputs)

            params, opt_state, state = update(
                params,
                state,
                opt_state,
                prior_mean,
                prior_cov,
                x_batch,
                y_batch,
                inducing_inputs,
                rng_key_train,
            )

        logging.T = time.time()
        logging.training_progress_ci(epoch, params, state, rng_key_test)

    if save:
        with open(f"{save_path}/params_pickle", "wb") as file:
            pickle.dump(params, file)
