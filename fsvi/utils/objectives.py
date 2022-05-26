from functools import partial
from typing import Tuple

import tree
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import jit, random
from jax.experimental import optimizers
from jax.tree_util import tree_flatten
import haiku as hk
from jax.scipy.stats.poisson import pmf as possion_pmf

from fsvi.utils import utils
from fsvi.utils import utils_linearization
from fsvi.utils.haiku_mod import (
    predicate_mean,
    predicate_var,
    predicate_batchnorm,
    partition_all_params,
    partition_params,
    partition_params_final_layer_bnn,
    gaussian_sample_pytree,
)
from fsvi.utils.utils_linearization import convert_predict_f_only_mean

dtype_default = jnp.float32
eps = 1e-6


class Objectives_hk:
    def __init__(
        self,
        architecture,
        model_type,
        feature_map_type,
        batch_normalization,
        batch_normalization_mod,
        apply_fn,
        predict_f,
        predict_f_deterministic,
        predict_y,
        predict_y_multisample,
        predict_y_multisample_jitted,
        output_dim,
        kl_scale: str,
        td_prior_scale,
        n_batches,
        predict_f_multisample,
        predict_f_multisample_jitted,
        noise_std,
        regularization,
        n_samples,
        full_cov,
        prior_type,
        stochastic_linearization,
        grad_flow_jacobian,
        stochastic_prior_mean,
        final_layer_variational,
        kl_sup,
        kl_sampled,
        n_marginals,
        params_init,
        full_ntk=False,
    ):
        self.architecture = architecture
        self.model_type = model_type
        self.feature_map_type = feature_map_type
        self.apply_fn = apply_fn
        self.predict_f = predict_f
        self.predict_f_deterministic = predict_f_deterministic
        self.predict_y = predict_y
        self.predict_f_multisample = predict_f_multisample
        self.predict_f_multisample_jitted = predict_f_multisample_jitted
        self.predict_y_multisample = predict_y_multisample
        self.predict_y_multisample_jitted = predict_y_multisample_jitted
        self.output_dim = output_dim
        self.kl_scale = kl_scale
        self.td_prior_scale = td_prior_scale
        self.regularization = regularization
        self.noise_std = noise_std
        self.n_batches = n_batches
        self.n_samples = n_samples
        self.full_cov = full_cov
        self.prior_type = prior_type
        self.stochastic_linearization = stochastic_linearization
        self.grad_flow_jacobian = grad_flow_jacobian
        self.final_layer_variational = final_layer_variational
        self.full_ntk = full_ntk
        self.kl_sup = kl_sup
        self.kl_sampled = kl_sampled
        self.n_marginals = n_marginals
        self.stochastic_prior_mean = stochastic_prior_mean
        self.params_init = params_init

        self.batch_normalization = batch_normalization
        self.batch_normalization_mod = batch_normalization_mod

        self.is_training = True
        self.update_state = True if self.batch_normalization_mod == "not_specified" else False
        self.direct_ntk = True

        if self.feature_map_type == "learned_nograd":
            assert self.grad_flow_jacobian == False
        elif self.feature_map_type == "learned_grad":
            assert self.grad_flow_jacobian == True

        if self.final_layer_variational:
            self.partition_params = partition_params_final_layer_bnn
        else:
            self.partition_params = partition_params

    @partial(jit, static_argnums=(0,10,))
    def objective_and_state(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        objective_fn,
    ):
        fixed_params = non_trainable_params[0]
        params_feature = non_trainable_params[1]

        if isinstance(inducing_inputs, list):
            prior_mean = inducing_inputs[1] * prior_mean
            inducing_inputs = inducing_inputs[0]
        if inducing_inputs is None:
            inducing_inputs = inputs

        params = hk.data_structures.merge(trainable_params, fixed_params)

        objective = objective_fn(
            params,
            params_feature,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            self.is_training,
        )

        # TODO: Clean up:
        if self.update_state:
            _inputs = inputs
            # _inputs = jnp.concatenate([inputs, inducing_inputs], axis=0)

            # update state with training input points
            state = self.apply_fn(
                params, state, rng_key, _inputs, rng_key, stochastic=True, is_training=self.is_training
            )[1]

            # update state with inducing input points
            # state = self.apply_fn(
            #     params, state, rng_key, inducing_inputs[:2], rng_key, stochastic=True, is_training=self.is_training
            # )[1]

        return objective, state

    @partial(jit, static_argnums=(0,))
    def accuracy(self, preds, targets):
        target_class = jnp.argmax(targets, axis=1)
        predicted_class = jnp.argmax(preds, axis=1)
        return jnp.mean(predicted_class == target_class)

    @partial(jit, static_argnums=(0,))
    def kl_gaussian(self, mean_q, mean_p, log_var_q, log_var_p, rng_key):
        mean_p = jnp.ones_like(mean_q) * mean_p
        cov_p = jnp.ones_like(log_var_q) * jnp.exp(log_var_p)
        cov_q = jnp.exp(log_var_q)
        kl = utils.kl_diag(mean_q, mean_p, cov_q, cov_p, self.kl_sampled, rng_key)
        return kl

    def _gaussian_log_likelihood(self, preds_f_samples, targets):
        targets = jnp.tile(targets, (preds_f_samples.shape[0], 1, 1))
        likelihood = tfd.Normal(preds_f_samples, self.noise_std)
        log_likelihood = jnp.sum(
            jnp.sum(jnp.mean(likelihood.log_prob(targets), 0), 0), 0
        )
        # preds_f_mean = preds_f_mean.reshape(-1, 1)
        # targets = targets.reshape(-1, 1)
        # likelihood = tfd.Normal(preds_f_mean, self.noise_std)
        # log_likelihood = -jnp.mean(jnp.sum((targets - preds_f_mean) ** 2, 1), 0)
        return log_likelihood

    def _crossentropy_log_likelihood(self, preds_f_samples, targets):
        log_likelihood = jnp.mean(
            jnp.sum(
                jnp.sum(
                    targets * jax.nn.log_softmax(preds_f_samples, axis=-1), axis=-1
                ),
                axis=-1,
            ),
            axis=0,
        )
        return log_likelihood

    def _parameter_kl(
        self,
        params,
        prior_mean,
        prior_cov,
        rng_key,
    ):
        # TODO: Move to separate function
        if self.stochastic_prior_mean != "not_specified":
            if self.stochastic_prior_mean[0] == "uniform":
                prior_mean = jax.random.uniform(
                    rng_key, shape=prior_mean.shape, dtype=dtype_default,
                    minval=prior_mean - self.stochastic_prior_mean[1], maxval=prior_mean + self.stochastic_prior_mean[1]
                )
            else:
                raise NotImplementedError

        params_mean, params_log_var, params_batchnorm = self.partition_params(params)
        params_mean_tp, _ = jax.flatten_util.ravel_pytree(params_mean)
        params_log_var_tp, _ = jax.flatten_util.ravel_pytree(params_log_var)
        kl = jnp.array(self.kl_gaussian(params_mean_tp, prior_mean, params_log_var_tp, jnp.log(prior_cov), rng_key)).sum()
        # Note: The below implementation is wrong, it only considers the first layer
        # params_mean_tp, _ = tree_flatten(params_mean)
        # params_log_var_tp, _ = tree_flatten(params_log_var)
        # kl = jnp.array(list(map(self.kl_gaussian, params_mean_tp, prior_mean, params_log_var_tp, jnp.log(prior_cov)))).sum()
        return kl

    def _function_kl(
        self, params, params_feature, state, prior_mean, prior_cov, inputs, inducing_inputs, rng_key,
    ) -> Tuple[jnp.ndarray, float]:
        """
        Evaluate the multi-output KL between the function distribution obtained by linearising BNN around
        params, and the prior function distribution represented by (`prior_mean`, `prior_cov`)

        @param inputs: used for computing scale, only the shape is used
        @param inducing_inputs: used for computing scale and function distribution used in KL

        @return:
            kl: scalar value of function KL
            scale: scale to multiple KL with
        """
        params_mean, params_log_var, params_deterministic = self.partition_params(params)
        params_feature_mean, params_feature_log_var, params_feature_deterministic = self.partition_params(params_feature)
        scale = compute_scale(self.kl_scale, inputs, inducing_inputs.shape[0], self.n_marginals)

        # mean, cov = self.linearize_fn(
        #     params_mean=params_mean,
        #     params_log_var=params_log_var,
        #     params_batchnorm=params_batchnorm,
        #     state=state,
        #     inducing_inputs=inducing_inputs,
        #     rng_key=rng_key,
        # )

        mean, cov = utils_linearization.bnn_linearized_predictive(
            self.apply_fn,
            self.predict_f,
            self.predict_f_deterministic,
            params_mean,
            params_log_var,
            params_deterministic,
            params_feature_mean,
            params_feature_log_var,
            params_feature_deterministic,
            state,
            inputs,
            inducing_inputs,
            rng_key,
            self.stochastic_linearization,
            self.full_ntk,
            self.is_training,
            self.grad_flow_jacobian,
            self.direct_ntk,
        )
        # TODO: Gaussian moment matching estimator, implement hyper for this
        # preds_f_samples, _, _ = self.predict_f_multisample_jitted(
        #     params, state, inducing_inputs, rng_key, self.n_samples, True,
        # )
        #
        # mean = jnp.mean(preds_f_samples, 0)
        # cov = jnp.square(jnp.std(preds_f_samples, 0))

        kl = utils.kl_divergence(
            mean,
            prior_mean,
            cov,
            prior_cov,
            self.output_dim,
            self.full_cov,
            self.prior_type,
            self.kl_sup,
            self.n_marginals,
            self.kl_sampled,
            rng_key,
        )

        return kl, scale

    def _nll_loss_regression(self, params, params_feature, state, inputs, targets, rng_key, is_training):
        preds_f_samples, _, _ = self.predict_f_multisample_jitted(
            params, params_feature, state, inputs, rng_key, self.n_samples, is_training,
        )
        log_likelihood = (
            self.gaussian_log_likelihood(preds_f_samples, targets) / targets.shape[0]
        )
        loss = -log_likelihood
        return loss, preds_f_samples

    def _nll_loss_classification(self, params, params_feature, state, inputs, targets, rng_key, is_training):
        preds_f_samples, _, _ = self.predict_f_multisample_jitted(
            params, params_feature, state, inputs, rng_key, self.n_samples, is_training,
        )
        log_likelihood = (
            self.crossentropy_log_likelihood(preds_f_samples, targets) / targets.shape[0]
        )
        loss = -log_likelihood
        return loss, preds_f_samples

    def _map_loss_regression(
        self,
        params,
        params_feature,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        is_training,
    ):
        negative_log_likelihood, preds_f_samples = self._nll_loss_regression(
            params, params_feature, state, inputs, targets, rng_key, is_training,
        )

        _params_mean, params_batchnorm, params_log_var, params_rest = partition_all_params(params)
        params_mean = _params_mean

        reg = jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x: jnp.square(x), params_mean))[0])

        # params_log_var = jax.lax.stop_gradient(jax.tree_map(lambda x: jnp.log((jnp.abs(x) * 0.01) ** 2), _params_mean))
        # params_mean_sample = gaussian_sample_pytree(_params_mean, params_log_var, rng_key)
        # params_mean = jax.lax.stop_gradient(_params_mean)
        # reg = jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_multimap(
        #     lambda x, y: jnp.square(x - y), params_mean_sample, params_eval
        # ))[0])

        loss = negative_log_likelihood + self.regularization * reg

        return loss

    def _map_loss_classification(
        self,
        params,
        params_feature,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        is_training,
    ):
        negative_log_likelihood, preds_f_samples = self._nll_loss_classification(
            params, params_feature, state, inputs, targets, rng_key, is_training,
        )

        _params_mean, params_batchnorm, params_log_var, params_rest = partition_all_params(params)
        params_mean = _params_mean

        reg = jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x: jnp.square(x), params_mean))[0])

        # params_log_var = jax.lax.stop_gradient(jax.tree_map(lambda x: jnp.log((jnp.abs(x) * 0.01) ** 2), _params_mean))
        # params_mean_sample = gaussian_sample_pytree(_params_mean, params_log_var, rng_key)
        # params_mean = jax.lax.stop_gradient(_params_mean)
        # reg = jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_multimap(
        #     lambda x, y: jnp.square(x - y), params_mean_sample, params_eval
        # ))[0])

        loss = negative_log_likelihood + self.regularization * reg

        return loss

    def _tdmap_loss_regression(
        self,
        params,
        params_feature,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        is_training,
    ):
        preds_f_train_samples, _, _ = self.predict_f_multisample_jitted(
            params, params_feature, state, inputs, rng_key, self.n_samples, is_training,
        )
        negative_log_likelihood = (-1) * self.gaussian_log_likelihood(preds_f_train_samples, targets)

        preds_f_inducing_samples, _, _ = self.predict_f_multisample_jitted(
            params, params_feature, state, inducing_inputs, rng_key, self.n_samples, is_training,
        )

        preds_f_train_mean = jnp.mean(preds_f_train_samples, axis=0)
        preds_f_inducing_mean = jnp.mean(preds_f_inducing_samples, axis=0)

        n_inducing = preds_f_inducing_mean.shape[0]
        n_inducing_prior = self.td_prior_scale
        weight = jax.random.poisson(rng_key, lam=n_inducing_prior)
        # weight = possion_pmf(k=n_inducing, mu=n_inducing, loc=0)

        reg_data = jnp.sum(jnp.square(preds_f_train_mean - prior_mean * jnp.ones_like(preds_f_train_mean)))
        reg_inducing = weight / n_inducing * jnp.sum(jnp.square(preds_f_inducing_mean - prior_mean))

        loss = (negative_log_likelihood + self.regularization * (reg_data + reg_inducing)) / inputs.shape[0]

        return loss

    def _tdmap_loss_classification(
        self,
        params,
        params_feature,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        is_training,
    ):
        preds_f_train_samples, _, _ = self.predict_f_multisample_jitted(
            params, params_feature, state, inputs, rng_key, self.n_samples, is_training,
        )
        negative_log_likelihood = (-1) * self.crossentropy_log_likelihood(preds_f_train_samples, targets)

        preds_f_inducing_samples, _, _ = self.predict_f_multisample_jitted(
            params, params_feature, state, inducing_inputs, rng_key, self.n_samples, is_training,
        )

        preds_f_train_mean = jnp.mean(preds_f_train_samples, axis=0)
        preds_f_inducing_mean = jnp.mean(preds_f_inducing_samples, axis=0)

        n_inducing = preds_f_inducing_mean.shape[0]
        n_inducing_prior = self.td_prior_scale
        weight = jax.random.poisson(rng_key, lam=n_inducing_prior)
        # weight = possion_pmf(k=n_inducing, mu=n_inducing, loc=0)

        reg_data = jnp.sum(jnp.square(preds_f_train_mean - prior_mean * jnp.ones_like(preds_f_train_mean)))
        reg_inducing = weight / n_inducing * jnp.sum(jnp.square(preds_f_inducing_mean - prior_mean))

        loss = (negative_log_likelihood + self.regularization * (reg_data + reg_inducing)) / inputs.shape[0]

        return loss

    def _elbo_mfvi_regression(
        self,
        params,
        params_feature,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        is_training,
    ):
        preds_f_samples, _, _ = self.predict_f_multisample_jitted(
            params, params_feature, state, inputs, rng_key, self.n_samples, is_training,
        )
        log_likelihood = self.gaussian_log_likelihood(preds_f_samples, targets)
        kl = self.parameter_kl(params, prior_mean, prior_cov, rng_key)

        scale = 1.0 / self.n_batches
        elbo = (log_likelihood - dtype_default(self.kl_scale) * scale * kl) / inputs.shape[0]

        return elbo, log_likelihood, kl, scale

    def _elbo_mfvi_classification(
        self,
        params,
        params_feature,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        is_training,
    ):
        preds_f_samples, _, _ = self.predict_f_multisample_jitted(
            params, params_feature, state, inputs, rng_key, self.n_samples, is_training,
        )
        log_likelihood = self.crossentropy_log_likelihood(preds_f_samples, targets)
        kl = self.parameter_kl(params, prior_mean, prior_cov, rng_key)

        scale = 1.0 / self.n_batches
        elbo = (log_likelihood - dtype_default(self.kl_scale) * scale * kl) / inputs.shape[0]

        return elbo, log_likelihood, kl, scale


    def _elbo_fsvi_regression(
        self,
        params,
        params_feature,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        is_training,
    ):
        preds_f_samples, _, _ = self.predict_f_multisample_jitted(
            params, params_feature, state, inputs, rng_key, self.n_samples, is_training,
        )
        log_likelihood = self.gaussian_log_likelihood(preds_f_samples, targets)
        kl, scale = self.function_kl(
            params, params_feature, state, prior_mean, prior_cov, inputs, inducing_inputs, rng_key,
        )

        elbo = (log_likelihood - scale * kl) / inputs.shape[0]

        return elbo, log_likelihood, kl, scale

    def _elbo_fsvi_classification(
        self,
        params,
        params_feature,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        is_training,
    ):
        preds_f_samples, _, _ = self.predict_f_multisample_jitted(
            params, params_feature, state, inputs, rng_key, self.n_samples, is_training,
        )
        log_likelihood = self.crossentropy_log_likelihood(preds_f_samples, targets)
        kl, scale = self.function_kl(
            params, params_feature, state, prior_mean, prior_cov, inputs, inducing_inputs, rng_key,
        )

        elbo = (log_likelihood - scale * kl) / inputs.shape[0]

        return elbo, log_likelihood, kl, scale

    def _elbo_tdvi_regression(
        self,
        params,
        params_feature,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        is_training,
    ):
        preds_f_samples, _, _ = self.predict_f_multisample_jitted(
            params, params_feature, state, inputs, rng_key, self.n_samples, is_training,
        )
        log_likelihood = self.gaussian_log_likelihood(preds_f_samples, targets)

        n_inducing_data = inducing_inputs.shape[0]
        permutation = jax.random.permutation(key=rng_key, x=inputs.shape[0])
        inputs_sample = inputs[permutation][:n_inducing_data]

        if not self.full_ntk:
            n_inducing_prior = self.td_prior_scale // inducing_inputs.shape[0]
            weight_inducing_data = inputs.shape[0] // n_inducing_data  # weight scaled up to mini-batch size
            weight_inducing_prior = jnp.max(jnp.array([jax.random.poisson(rng_key, lam=n_inducing_prior), 1]), 0)

            kl_data, _ = self.function_kl(
                params, params_feature, state, prior_mean, prior_cov, inputs_sample, inputs_sample, rng_key,
            )
            kl_prior, _ = self.function_kl(
                params, params_feature, state, prior_mean, prior_cov, inducing_inputs, inducing_inputs, rng_key,
            )

            kl = kl_data * weight_inducing_data + kl_prior * weight_inducing_prior
        else:
            assert prior_mean.shape[0] == inducing_inputs.shape[0]

            kl, _ = self.function_kl(
                params, params_feature, state, prior_mean, prior_cov, inducing_inputs, inducing_inputs, rng_key,
            )

        params_mean, params_batchnorm, params_log_var, params_rest = partition_all_params(params)
        reg = jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x: jnp.square(x), params_mean))[0])

        llk_scale = 1  #self.n_batches
        scale = 0.0
        elbo = (llk_scale * log_likelihood - kl - self.regularization * reg) / inputs.shape[0]


        return elbo, log_likelihood, kl, scale

    def _elbo_tdvi_classification(
        self,
        params,
        params_feature,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        is_training,
    ):
        preds_f_samples, _, _ = self.predict_f_multisample_jitted(
            params, params_feature, state, inputs, rng_key, self.n_samples, is_training,
        )
        log_likelihood = self.crossentropy_log_likelihood(preds_f_samples, targets)

        n_inducing_data = inducing_inputs.shape[0]
        n_inducing_prior = self.td_prior_scale // inducing_inputs.shape[0]
        weight_inducing_data = inputs.shape[0] // n_inducing_data  # weight scaled up to mini-batch size
        # weight_inducing_data = n_inducing_data
        weight_inducing_prior = jnp.max(jnp.array([jax.random.poisson(rng_key, lam=n_inducing_prior), 1]), 0)

        permutation = jax.random.permutation(key=rng_key, x=inputs.shape[0])
        inputs_sample = inputs[permutation][:n_inducing_data]

        kl_data, _ = self.function_kl(
            params, params_feature, state, prior_mean, prior_cov, inputs_sample, inputs_sample, rng_key,
        )
        kl_prior, _ = self.function_kl(
            params, params_feature, state, prior_mean, prior_cov, inducing_inputs, inducing_inputs, rng_key,
        )

        kl = kl_prior * weight_inducing_prior + kl_data * weight_inducing_data
        # kl = kl_prior * weight_inducing_prior
        # kl = kl_data * weight_inducing_data

        params_mean, params_batchnorm, params_log_var, params_rest = partition_all_params(params)
        reg = jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x: jnp.square(x), params_mean))[0])

        llk_scale = 1  #self.n_batches
        scale = 0.0
        elbo = (llk_scale * log_likelihood - kl - self.regularization * reg) / inputs.shape[0]

        # TODO: Delete later:
        # params_mean, params_log_var, params_deterministic = self.partition_params(params)
        # params_fl = hk.data_structures.merge(params_mean, params_log_var)
        # kl_params = 0
        # scale_params = 1.0 / self.n_batches
        # self.parameter_kl(
        #     params_fl,
        #     0.,
        #     5.,
        #     rng_key,
        # )
        # elbo = (log_likelihood - scale * kl - scale_params * kl_params) / inputs.shape[0]

        return elbo, log_likelihood, kl, scale

    @partial(jit, static_argnums=(0,))
    def gaussian_log_likelihood(self, preds_f_samples, targets):
        return self._gaussian_log_likelihood(preds_f_samples, targets)

    @partial(jit, static_argnums=(0,))
    def crossentropy_log_likelihood(self, preds_f_samples, targets):
        return self._crossentropy_log_likelihood(preds_f_samples, targets)

    @partial(jit, static_argnums=(0,))
    def function_kl(
        self, params, params_feature, state, prior_mean, prior_cov, inputs, inducing_inputs, rng_key,
    ):
        return self._function_kl(
            params=params,
            params_feature=params_feature,
            state=state,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            inputs=inputs,
            inducing_inputs=inducing_inputs,
            rng_key=rng_key,
        )

    @partial(jit, static_argnums=(0,))
    def parameter_kl(
        self,
        params,
        prior_mean,
        prior_cov,
        rng_key,
    ):
        return self._parameter_kl(
            params,
            prior_mean,
            prior_cov,
            rng_key,
        )

    @partial(jit, static_argnums=(0,))
    def nll_loss_regression(self, trainable_params, non_trainable_params, state, inputs, targets, rng_key):
        objective, state = self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            inputs,
            targets,
            rng_key,
            self._nll_loss_regression
        )

        return objective, state

    @partial(jit, static_argnums=(0,))
    def nll_loss_classification(self, trainable_params, non_trainable_params, state, inputs, targets, rng_key):
        objective, state = self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            inputs,
            targets,
            rng_key,
            self._nll_loss_classification
        )

        return objective, state

    @partial(jit, static_argnums=(0,))
    def map_loss_regression(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
    ):
        objective, state = self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            self._map_loss_regression,
        )

        return objective, state

    @partial(jit, static_argnums=(0,))
    def map_loss_classification(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
    ):
        objective, state = self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            self._map_loss_classification,
        )

        return objective, state

    @partial(jit, static_argnums=(0,))
    def tdmap_loss_regression(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
    ):
        objective, state = self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            self._tdmap_loss_regression,
        )

        return objective, state

    @partial(jit, static_argnums=(0,))
    def tdmap_loss_classification(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
    ):
        objective, state = self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            self._tdmap_loss_classification,
        )

        return objective, state

    @partial(jit, static_argnums=(0,))
    def nelbo_mfvi_regression(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
    ):
        (elbo, log_likelihood, kl, scale), state = self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            self._elbo_mfvi_regression,
        )

        return -elbo, state

    @partial(jit, static_argnums=(0,))
    def nelbo_mfvi_classification(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
    ):
        (elbo, log_likelihood, kl, scale), state = self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            self._elbo_mfvi_classification,
        )

        return -elbo, state

    @partial(jit, static_argnums=(0,))
    def nelbo_fsvi_regression(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
    ):
        (elbo, log_likelihood, kl, scale), state = self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            self._elbo_fsvi_regression,
        )

        return -elbo, state

    @partial(jit, static_argnums=(0,))
    def nelbo_fsvi_classification(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
    ):
        (elbo, log_likelihood, kl, scale), state = self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            self._elbo_fsvi_classification,
        )

        return -elbo, state
    
    
    @partial(jit, static_argnums=(0,))
    def nelbo_tdvi_regression(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
    ):
        (elbo, log_likelihood, kl, scale), state = self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            self._elbo_tdvi_regression,
        )

        return -elbo, state

    @partial(jit, static_argnums=(0,))
    def nelbo_tdvi_classification(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
    ):
        (elbo, log_likelihood, kl, scale), state = self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            self._elbo_tdvi_classification,
        )

        return -elbo, state


@partial(jit, static_argnums=(0,3,))
def compute_scale(kl_scale: str, inputs: jnp.ndarray, n_inducing_inputs: int, n_marginals: int) -> float:
    if kl_scale == "none":
        scale = 1.0
    elif kl_scale == "equal":
        scale = inputs.shape[0] / n_inducing_inputs
    elif kl_scale == "normalized":
        if n_marginals > 1:
            scale = 1.0 / (n_inducing_inputs // n_marginals)
        else:
            scale = 1.0 / n_inducing_inputs
    else:
        scale = dtype_default(kl_scale)
    return scale
