from functools import partial
from typing import List, Tuple, Callable, Union, Sequence, Dict

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import sklearn
import tree

from fsvi.models.networks import CNN, Model
from fsvi.models.networks import MLP as MLP
from fsvi.utils import utils, utils_linearization
from fsvi.utils.haiku_mod import predicate_mean, predicate_var, predicate_batchnorm, partition_params
from fsvi.utils.objectives import Objectives_hk as Objectives

classification_datasets = [
    "mnist",
    "notmnist",
    "fashionmnist",
    "cifar10",
    "cifar10_noaugmentation",
    "svhn",
    "two_moons",
]
continual_learning_datasets = [
    "pmnist",
    "smnist",
    "smnist_sh",
    "pfashionmnist",
    "sfashionmnist",
    "sfashionmnist_sh",
    "cifar_full",
    "cifar_small",
    "cifar_small_sh",
    "cifar100_sh",
    "toy",
    "toy_sh",
    "toy_reprod",
    "omniglot",
]
classification_datasets.extend(
    [f"continual_learning_{ds}" for ds in continual_learning_datasets]
)
regression_datasets = ["uci", "offline_rl", "snelson", "oat1d", "subspace_inference","ihdp"]

dtype_default = jnp.float32


class Training:
    def __init__(
        self,
        task: str,
        data_training,
        model_type: str,
        optimizer: str,
        optimizer_var: str,
        inducing_input_type: str,
        inducing_input_ood_data: str,
        prior_type,
        prior_covs: list,
        architecture,
        no_final_layer_bias: bool,
        activation: str,
        learning_rate,
        learning_rate_var,
        schedule,
        dropout_rate,
        batch_normalization: bool,
        batch_normalization_mod: bool,
        input_shape: List[int],
        output_dim: int,
        full_ntk: bool,
        regularization,
        kl_scale,
        td_prior_scale,
        stochastic_linearization: bool,
        grad_flow_jacobian: bool,
        final_layer_variational: bool,
        fixed_inner_layers_variational_var: bool,
        start_var_opt: int,
        extra_linear_layer: bool,
        feature_map_jacobian: bool,
        feature_map_jacobian_train_only: bool,
        feature_map_type: bool,
        init_logvar_minval: float,
        init_logvar_maxval: float,
        init_logvar_lin_minval: float,
        init_logvar_lin_maxval: float,
        init_logvar_conv_minval: float,
        init_logvar_conv_maxval: float,
        perturbation_param: float,
        full_cov,
        kl_sup,
        kl_sampled,
        n_marginals,
        n_condition,
        stochastic_prior_mean,
        n_samples,
        n_train: int,
        epochs,
        batch_size,
        n_batches,
        inducing_inputs_bound: List[int],
        n_inducing_inputs: int,
        noise_std,
        map_initialization,
        momentum,
        momentum_var,
        **kwargs,
    ):
        """

        @param task: examples: continual_learning_pmnist, continual_learning_sfashionmnist
        @param n_inducing_inputs: number of inducing points to draw from each task
        @param output_dim: the task-specific number of output dimensions
        """
        self.task = task
        self.data_training = data_training
        self.model_type = model_type
        self.optimizer = optimizer
        self.optimizer_var = optimizer_var
        self.momentum = momentum
        self.momentum_var = momentum_var
        self.inducing_input_type = inducing_input_type
        self.inducing_input_ood_data = inducing_input_ood_data
        self.prior_type = prior_type
        self.prior_covs = prior_covs
        self.architecture = architecture
        self.no_final_layer_bias = no_final_layer_bias
        self.activation = activation
        self.learning_rate = learning_rate
        self.learning_rate_var = learning_rate_var
        self.schedule = schedule
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.batch_normalization_mod = batch_normalization_mod
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.full_ntk = full_ntk
        self.regularization = regularization
        self.kl_scale = kl_scale
        self.td_prior_scale = td_prior_scale
        self.stochastic_linearization = stochastic_linearization
        self.grad_flow_jacobian = grad_flow_jacobian
        self.final_layer_variational = final_layer_variational
        self.fixed_inner_layers_variational_var = fixed_inner_layers_variational_var
        self.start_var_opt = start_var_opt
        self.extra_linear_layer = extra_linear_layer
        self.feature_map_jacobian = feature_map_jacobian
        self.feature_map_jacobian_train_only = feature_map_jacobian_train_only
        self.feature_map_type = feature_map_type
        self.init_logvar_minval = init_logvar_minval
        self.init_logvar_maxval = init_logvar_maxval
        self.init_logvar_lin_minval = init_logvar_lin_minval
        self.init_logvar_lin_maxval = init_logvar_lin_maxval
        self.init_logvar_conv_minval = init_logvar_conv_minval
        self.init_logvar_conv_maxval = init_logvar_conv_maxval
        self.perturbation_param = perturbation_param
        self.full_cov = full_cov
        self.kl_sup = kl_sup
        self.kl_sampled = kl_sampled
        self.n_marginals = n_marginals
        self.n_condition = n_condition
        self.n_samples = n_samples
        self.n_train = n_train
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.inducing_inputs_bound = inducing_inputs_bound
        self.n_inducing_inputs = n_inducing_inputs
        self.noise_std = noise_std
        self.task_id = 0

        self.x_condition = None

        self.map_initialization = map_initialization

        self.resnet = True if "resnet" in self.model_type else False

        self.dropout = "dropout" in self.model_type
        if not self.dropout and self.dropout_rate > 0:
            raise ValueError("Dropout rate not zero in non-dropout model.")

        if prior_type == 'bnn_induced':
            self.stochastic_linearization_prior = True
        elif prior_type == 'blm_induced':
            self.stochastic_linearization_prior = False
        else:
            self.stochastic_linearization_prior = False

        if stochastic_prior_mean != "not_specified":
            self.stochastic_prior_mean = [
                stochastic_prior_mean.split("_", 1)[0],
                dtype_default(stochastic_prior_mean.split("_", 1)[1])
            ]
        else:
            self.stochastic_prior_mean = stochastic_prior_mean

        print(f"\n"
              f"MAP initialization: {self.map_initialization}")
        print(f"Full NTK computation: {self.full_ntk}")
        print(f"Variational final layer: {self.final_layer_variational}")
        print(f"Stochastic linearization (posterior): {self.stochastic_linearization}")
        print(f"Stochastic linearization (prior): {self.stochastic_linearization_prior}")
        print(f"Gradient flow through Jacobian evaluation: {self.grad_flow_jacobian}"
              f"\n")

    def initialize_model(
        self,
        rng_key,
        x_train,
        x_ood,
    ):
        if self.batch_normalization_mod != "not_specified":
            if len(x_ood) != 0:
                x_ood = jnp.concatenate(x_ood, axis=0)
                permutation = jax.random.permutation(key=rng_key, x=x_ood.shape[0])
                x_ood_permuted = x_ood[permutation, :]
                permutation = jax.random.permutation(key=rng_key, x=x_train.shape[0])
                x_train_permuted = x_train[permutation, :]
                self.x_condition = jnp.concatenate([x_train_permuted[0:int(self.n_condition / 2)], x_ood_permuted[0:int(self.n_condition / 2)]], axis=0)
            else:
                permutation = jax.random.permutation(key=rng_key, x=x_train.shape[0])
                x_train_permuted = x_train[permutation, :]
                self.x_condition = x_train_permuted[0:self.n_condition]

        model = self._compose_model()
        # INITIALIZE NETWORK STATE + PARAMETERS
        x_init = jnp.ones(self.input_shape)
        if 'mlp' in self.model_type:
            x_init = x_init.reshape([x_init.shape[0], -1])

        if "vit" in self.architecture:
            init_fn, apply_fn = model.forward.init_fn, model.forward.apply
            params_init = init_fn(rng_key, x_init, model.stochastic_parameters, is_training=False)
            # params_count = sum(p.size for p in jax.tree_flatten(params_init)[0])
            state = {}
        else:
            init_fn, apply_fn = model.forward
            params_init, state = init_fn(
                rng_key, x_init, rng_key, model.stochastic_parameters, is_training=True
            )

        # Initializes exponential moving average
        n_eval = 500
        for i in range(10):
            _x_train_init = x_train[i*n_eval:(i+1)*n_eval]
            _, state = apply_fn(
                params_init, state, rng_key, _x_train_init, rng_key, model.stochastic_parameters, is_training=True
            )

        if self.map_initialization:
            params_init = self._pretraining_initialization(params_init)

        model.params_init = params_init
        model.params_eval_carry = params_init

        return model, init_fn, apply_fn, state, params_init

    def initialize_optimization(
        self,
        model,
        apply_fn: Callable,
        params_init: hk.Params,
        state: hk.State,
        rng_key: jnp.ndarray,
    ) -> Tuple[
        optax.GradientTransformation,
        Union[optax.OptState, Sequence[optax.OptState]],
        Callable,
        Callable,
        Objectives,
        Callable,
        Callable,
        Callable,
        Callable,
        Callable,
        str,
    ]:
        opt_mu = self._compose_optimizer(self.optimizer, self.learning_rate, self.momentum)
        if self.optimizer_var == "not_specified":
            opt = opt_mu
        else:
            opt_var = self._compose_optimizer(self.optimizer_var, self.learning_rate_var, self.momentum_var)
            opt = [opt_mu, opt_var]

        params_init_mu, params_init_logvar, _ = partition_params(params_init)

        if self.optimizer_var == "not_specified":
            opt_state = opt.init(params_init)
        else:
            opt_mu_state = opt_mu.init(params_init_mu)
            opt_var_state = opt_var.init(params_init_logvar)
            opt_state = [opt_mu_state, opt_var_state]

        get_trainable_params = self.get_trainable_params_fn(params_init)
        get_variational_and_model_params = self.get_params_partition_fn(params_init)

        # # FIXME: doesn't currently seem to work
        # def _pred_fn(apply_fn, params, state, inputs, rng_key, n_samples):
        #     rng_key, subkey = jax.random.split(rng_key)
        #     preds_samples = jnp.expand_dims(apply_fn(params, state, None, inputs, rng_key, stochastic=False, is_training=True)[0], 0)
        #     for i in range(n_samples - 1):
        #         rng_key, subkey = jax.random.split(rng_key)
        #         preds_samples = jnp.concatenate(
        #             (preds_samples, jnp.expand_dims(apply_fn(params, state, None, inputs, rng_key, stochastic=False, is_training=True)[0], 0)), 0)
        #     return preds_samples
        #
        # pred_fn = jax.jit(partial(_pred_fn,
        #     apply_fn=apply_fn,
        #     state=state,
        #     n_samples=self.n_samples,
        # ))

        prediction_type = decide_prediction_type(self.data_training)
        objective = self._compose_objective(
            model=model, apply_fn=apply_fn, state=state, rng_key=rng_key, params_init=params_init,
        )
        # LOSS
        loss, kl_evaluation = self._compose_loss(
            prediction_type=prediction_type, metrics=objective
        )
        # EVALUATION
        (
            log_likelihood_evaluation,
            nll_grad_evaluation,
            task_evaluation,
        ) = self._compose_evaluation_metrics(
            prediction_type=prediction_type, metrics=objective
        )

        return (
            opt,
            opt_state,
            get_trainable_params,
            get_variational_and_model_params,
            objective,
            loss,
            kl_evaluation,
            log_likelihood_evaluation,
            nll_grad_evaluation,
            task_evaluation,
            prediction_type,
        )

    def _compose_model(self) -> Model:
        if "mlp" in self.model_type:
            network_class = MLP
        elif "cnn" or "resnet" in self.model_type:
            network_class = CNN
        else:
            raise ValueError("Invalid network type.")

        stochastic_parameters = "mfvi" in self.model_type or "fsvi" in self.model_type or "tdvi" in self.model_type

        # DEFINE NETWORK
        model = network_class(
            architecture=self.architecture,
            no_final_layer_bias=self.no_final_layer_bias,
            output_dim=self.output_dim,
            activation_fn=self.activation,
            regularization=self.regularization,
            stochastic_parameters=stochastic_parameters,
            final_layer_variational=self.final_layer_variational,
            fixed_inner_layers_variational_var=self.fixed_inner_layers_variational_var,
            extra_linear_layer=self.extra_linear_layer,
            feature_map_jacobian=self.feature_map_jacobian,
            feature_map_jacobian_train_only=self.feature_map_jacobian_train_only,
            feature_map_type=self.feature_map_type,
            dropout=self.dropout,
            dropout_rate=self.dropout_rate,
            resnet=self.resnet,
            batch_normalization=self.batch_normalization,
            batch_normalization_mod=self.batch_normalization_mod,
            x_condition=self.x_condition,
            init_logvar_minval=self.init_logvar_minval,
            init_logvar_maxval=self.init_logvar_maxval,
            init_logvar_lin_minval=self.init_logvar_lin_minval,
            init_logvar_lin_maxval=self.init_logvar_lin_maxval,
            init_logvar_conv_minval=self.init_logvar_conv_minval,
            init_logvar_conv_maxval=self.init_logvar_conv_maxval,
            perturbation_param=self.perturbation_param,
        )
        return model

    def _pretraining_initialization(self, params_init):
        if "vit" not in self.architecture:
            params_log_var_init = hk.data_structures.filter(predicate_var, params_init)

            if "fashionmnist" in self.task:
                if "resnet" not in self.architecture:
                    # filename_params = "saved_models/fashionmnist/map/params_pickle_map_fashionmnist"
                    filename_params = "/scratch-ssd/ANON/deployment/testing/projects/function-space-variational-inference/params_pickle_fmnist_01"
                else:
                    filename_params = "/scratch-ssd/ANON/deployment/testing/projects/function-space-variational-inference/params_pickle_fmnist_resnet"
                    filename_state = "/scratch-ssd/ANON/deployment/testing/projects/function-space-variational-inference/state_pickle_fmnist_resnet"
            elif "cifar" in self.task:
                if "resnet" not in self.architecture:
                    filename_params = "saved_models/cifar10/map/params_pickle_map_cifar10"
                else:
                    filename_params = "saved_models/cifar10/map/params_pickle_map_cifar10_resnet_01"
                    filename_state = "saved_models/cifar10/map/state_pickle_map_cifar10_resnet_01"
                    # filename_params = "saved_models/cifar10/fsvi/params_pickle_fsvi_cifar10_resnet_linear_model_01"
                    # filename_state = "saved_models/cifar10/fsvi/state_pickle_fsvi_cifar10_resnet_linear_model_01"
            else:
                raise ValueError("MAP parameter file not found.")

            # TODO: use absolute path instead of letting it depend on working directory?
            params_trained = np.load(filename_params, allow_pickle=True)
            if "resnet" in self.architecture:
                state = np.load(filename_state, allow_pickle=True)

            params_mean_trained = hk.data_structures.filter(predicate_mean, params_trained)
            params_batchnorm_trained = hk.data_structures.filter(
                predicate_batchnorm, params_trained
            )

            params_init = hk.data_structures.merge(
                params_mean_trained, params_log_var_init, params_batchnorm_trained
            )
        else:
            filename_params = "/scratch-ssd/ANON/deployment/testing/projects/function-space-variational-inference/params_pickle_vit_01"
            params_init = np.load(filename_params, allow_pickle=True)

        return params_init

    def _compose_optimizer(self, optimizer, learning_rate, momentum) -> optax.GradientTransformation:
        if "adam" in optimizer:
            # epoch_points = self.optimizer.split("_")[1:]
            # for i in range(len(epoch_points)):
            #     epoch_points[i] = int(epoch_points[i])
            # epoch_points = (jnp.array(epoch_points) * self.n_batches).tolist()
            #
            # def schedule_fn(learning_rate, n_batches):
            #     # return piecewise_constant_schedule(learning_rate, epoch_points, 0.5)
            #     return optax.polynomial_schedule(
            #         init_value=learning_rate,
            #         end_value=1e-5,
            #         power=0.95,
            #         transition_steps=epoch_points[-1]
            #     )
            # schedule_fn_final = schedule_fn(self.learning_rate, self.n_batches)
            #
            # def make_adam_optimizer():
            #     return optax.chain(
            #         optax.scale_by_schedule(schedule_fn_final),
            #         optax.scale_by_adam(),
            #         optax.scale(-self.learning_rate),
            #     )
            #
            # opt = make_adam_optimizer()
            opt = optax.adam(learning_rate)
        elif "sgd" in optimizer:
            epoch_points = optimizer.split("_")[1:]
            for i in range(len(epoch_points)):
                epoch_points[i] = int(epoch_points[i])
            epoch_points = (jnp.array(epoch_points) * self.n_batches).tolist()

            if "piecewiseconstant" in self.schedule:

                lr_schedule_param = dtype_default(self.schedule.split("_", 1)[1].split("_", 1)[0])

                def schedule_fn(learning_rate, epoch_points, lr_schedule_param):
                    return piecewise_constant_schedule(learning_rate, epoch_points, lr_schedule_param)

                schedule_fn_final = schedule_fn(learning_rate, epoch_points, lr_schedule_param)

            elif "linear" in self.schedule:
                scale = self.regularization if self.regularization != 0.0 else 1.0

                init_value = self.learning_rate * scale

                def schedule_fn(init_value, scale, n_batches):
                    return linear_schedule(init_value, scale, n_batches)

                schedule_fn_final = schedule_fn(init_value, scale, self.n_batches)

            elif "inv_sqrt" in self.schedule:
                scale = self.regularization if self.regularization != 0.0 else 1.0

                init_value = self.learning_rate * scale

                def schedule_fn(init_value, scale, n_batches):
                    return inv_sqrt_schedule(init_value, scale, n_batches)

                schedule_fn_final = schedule_fn(init_value, scale, self.n_batches)

            else:
                raise ValueError("No learning rate schedule type specified for SGD.")


            def make_sgd_optimizer():
                if momentum != 0:
                    return optax.chain(
                        optax.trace(decay=momentum, nesterov=False),
                        optax.scale_by_schedule(schedule_fn_final),
                        optax.scale(-1),
                    )
                else:
                    return optax.chain(
                        optax.scale_by_schedule(schedule_fn_final),
                        optax.scale(-1),
                    )

            opt = make_sgd_optimizer()
        # else:
        #     raise ValueError("No optimizer specified.")
        return opt

    def _compose_loss(
        self, prediction_type: str, metrics: Objectives
    ) -> Tuple[Callable, Callable]:
        assert "continual_learning" not in self.task, "This method is deprecated for continual learning"
        if "fsvi" in self.model_type:
            if prediction_type == "classification":
                loss = metrics.nelbo_fsvi_classification
            if prediction_type == "regression":
                loss = metrics.nelbo_fsvi_regression
            kl_evaluation = metrics.function_kl
        elif "tdvi" in self.model_type:
            if prediction_type == "classification":
                loss = metrics.nelbo_tdvi_classification
            if prediction_type == "regression":
                loss = metrics.nelbo_tdvi_regression
            kl_evaluation = metrics.function_kl
        elif "mfvi" in self.model_type:
            if prediction_type == "classification":
                loss = metrics.nelbo_mfvi_classification
            if prediction_type == "regression":
                loss = metrics.nelbo_mfvi_regression
            kl_evaluation = metrics.parameter_kl
        elif "map" in self.model_type or "dropout" in self.model_type:
            if prediction_type == "classification":
                if "tdmap" in self.model_type:
                    loss = metrics.tdmap_loss_classification
                else:
                    loss = metrics.map_loss_classification
            elif prediction_type == "regression":
                if "tdmap" in self.model_type:
                    loss = metrics.tdmap_loss_regression
                else:
                    loss = metrics.map_loss_regression
            kl_evaluation = None
        else:
            raise ValueError("No loss specified.")
        return loss, kl_evaluation

    def _compose_objective(self, model, apply_fn, state, rng_key, params_init) -> Objectives:
        # `linearize_fn` is a function that takes in params_mean, params_log_var, params_batchnorm,
        # inducing_inputs, rng_key, state, and returns mean and covariance of inducing_inputs

        metrics = Objectives(
            architecture=self.architecture,
            model_type=self.model_type,
            feature_map_type=self.feature_map_type,
            batch_normalization=self.batch_normalization,
            batch_normalization_mod=self.batch_normalization_mod,
            apply_fn=apply_fn,
            predict_f=model.predict_f,
            predict_f_deterministic=model.predict_f_deterministic,
            predict_y=model.predict_y,
            predict_f_multisample=model.predict_f_multisample,
            predict_f_multisample_jitted=model.predict_f_multisample_jitted,
            predict_y_multisample=model.predict_y_multisample,
            predict_y_multisample_jitted=model.predict_y_multisample_jitted,
            regularization=self.regularization,
            kl_scale=self.kl_scale,
            td_prior_scale=self.td_prior_scale,
            full_cov=self.full_cov,
            n_samples=self.n_samples,
            n_batches=self.n_batches,
            output_dim=self.output_dim,
            noise_std=self.noise_std,
            prior_type=self.prior_type,
            stochastic_linearization=self.stochastic_linearization,
            grad_flow_jacobian=self.grad_flow_jacobian,
            stochastic_prior_mean=self.stochastic_prior_mean,
            final_layer_variational=self.final_layer_variational,
            full_ntk=self.full_ntk,
            kl_sup=self.kl_sup,
            kl_sampled=self.kl_sampled,
            n_marginals=self.n_marginals,
            params_init=params_init,
        )
        return metrics

    def _compose_evaluation_metrics(
        self, prediction_type: str, metrics: Objectives
    ) -> Tuple[Callable, Callable, Callable]:
        assert "continual_learning" not in self.task, "This method is deprecated for continual learning"
        if prediction_type == "classification":
            nll_grad_evaluation = metrics.nll_loss_classification
            task_evaluation = metrics.accuracy
            log_likelihood_evaluation = metrics._crossentropy_log_likelihood
        elif prediction_type == "regression":
            task_evaluation = None  # TODO: implement MSE evaluation
            nll_grad_evaluation = metrics.nll_loss_regression
            log_likelihood_evaluation = metrics._gaussian_log_likelihood
        else:
            raise ValueError(f"Unrecognized prediction_type: {prediction_type}")
        return log_likelihood_evaluation, nll_grad_evaluation, task_evaluation

    def get_inducing_input_fn(
            self,
            x_ood=[],
            y_ood=[],
            rng_key=None,
    ):
        if len(x_ood) != 0:
            x_oods = jnp.concatenate(x_ood, axis=0)
            y_oods = jnp.concatenate(y_ood, axis=0)
            permutation = jax.random.permutation(key=rng_key, x=x_oods.shape[0])

            x_oods_permuted = x_oods[permutation]
            y_oods_permuted = y_oods[permutation]
        else:
            x_oods_permuted = None
            y_oods_permuted = None

        # if self.inducing_input_type == "ood_rand" and len(x_ood) > 1:
        #     raise AssertionError("Inducing point type 'ood_rand' only works if one OOD set is specified.")
        def inducing_input_fn(x_batch, rng_key, n_inducing_inputs=None):
            if n_inducing_inputs is None:
                n_inducing_inputs = self.n_inducing_inputs
            return utils.select_inducing_inputs(
                n_inducing_inputs=n_inducing_inputs,
                n_marginals=self.n_marginals,
                inducing_input_type=self.inducing_input_type,
                inducing_inputs_bound=self.inducing_inputs_bound,
                input_shape=self.input_shape,
                x_batch=x_batch,
                x_ood=x_oods_permuted,
                y_ood=y_oods_permuted,
                n_train=self.n_train,
                rng_key=rng_key,
            )
        return inducing_input_fn

    def get_prior_fn(
        self,
        apply_fn: Callable,
        predict_f: Callable,
        predict_f_deterministic: Callable,
        state: hk.State,
        params: hk.Params,
        prior_mean: str,
        prior_cov: str,
        rng_key,
        prior_type,
        task_id,
        jit_prior=True,
        identity_cov=False,
    ) -> Tuple[
        Callable[[jnp.ndarray], List[jnp.ndarray]],
    ]:
        if "fsvi" in self.model_type or "tdvi" in self.model_type:
            assert "continual_learning" not in self.task, "This method is deprecated for continual learning"
            if prior_type == "bnn_induced" or prior_type == "blm_induced":
                rng_key0, _ = jax.random.split(rng_key)

                params_prior = get_prior_params(
                    params_init=params,
                    prior_mean=prior_mean,
                    prior_cov=prior_cov,
                    batch_normalization=self.batch_normalization,
                )

                # prior_fn is a function of inducing_inputs and params
                prior_fn = lambda inducing_inputs, model_params, rng_key, prior_cov, state: partial(  # params args are unused
                    utils_linearization.induced_prior_fn_v0,
                    apply_fn=apply_fn,
                    predict_f=predict_f,
                    predict_f_deterministic=predict_f_deterministic,
                    params=params_prior,
                    state=state,
                    rng_key=rng_key0,
                    task_id=task_id,
                    n_inducing_inputs=self.n_inducing_inputs,
                    architecture=self.architecture,
                    stochastic_linearization=self.stochastic_linearization_prior,
                    full_ntk=self.full_ntk,
                )(inducing_inputs=inducing_inputs, params=params_prior)
                if jit_prior and not identity_cov:
                    prior_fn = jax.jit(prior_fn)

            elif prior_type == "rbf":
                prior_mean = jnp.ones(self.n_inducing_inputs) * prior_mean
                prior_fn = lambda inducing_inputs, model_params, rng_key, prior_cov, state: [
                    prior_mean,
                    sklearn.metrics.pairwise.rbf_kernel(
                        inducing_inputs.reshape([inducing_inputs.shape[0], -1]), gamma=None
                    )
                    * prior_cov,
                ]

            elif prior_type == "fixed":
                scale = 1.  # TODO: Add as config arg
                def prior_fn(inducing_inputs, model_params, rng_key, prior_cov, state):
                    if isinstance(prior_cov, list):
                        prior_cov_fn = lambda rng_key, prior_cov: jax.random.choice(rng_key, jnp.array(prior_cov))
                    else:
                        prior_cov_fn = lambda rng_key, prior_cov: prior_cov

                    prior_pred = [
                        jnp.concatenate(
                            [
                                jnp.ones([1,1]) * dtype_default(prior_mean),
                                jnp.ones([1,1]) * dtype_default(prior_mean)
                            ], 1).repeat(inducing_inputs.shape[0] // 2, axis=0).flatten(),
                        jnp.concatenate(
                            [
                                jnp.ones([1,1]) * prior_cov_fn(rng_key, dtype_default(prior_cov)),
                                jnp.ones([1,1]) * prior_cov_fn(rng_key, dtype_default(prior_cov)) * scale  # prior on training points
                            ], 1).repeat(inducing_inputs.shape[0] // 2, axis=0).flatten()
                    ]

                    return prior_pred

            elif prior_type == "map_mean":
                prior_mean = partial(
                    predict_f_deterministic, params=params, state=state, rng_key=rng_key
                )
                prior_cov = jnp.ones(self.n_inducing_inputs) * prior_cov
                prior_fn = lambda inducing_inputs, model_params, rng_key, proir_cov, state: [
                    prior_mean(inputs=inducing_inputs),
                    prior_cov,
                ]

                if jit_prior:
                    prior_fn = jax.jit(prior_fn)

            elif prior_type == "map_induced":
                rng_key0, _ = jax.random.split(rng_key)
                params_prior = params
                state_prior = state

                def prior_fn(inducing_inputs, model_params, rng_key, prior_cov, state):
                    return partial(
                        utils_linearization.induced_prior_fn_v0,
                        apply_fn=apply_fn,
                        state=state_prior,
                        rng_key=rng_key0,
                        task_id=task_id,
                        n_inducing_inputs=self.n_inducing_inputs,
                        architecture=self.architecture,
                        stochastic_linearization=self.stochastic_linearization_prior,
                        full_ntk=self.full_ntk,
                    )(inducing_inputs=inducing_inputs, params=params_prior)
                if jit_prior and not identity_cov:
                    prior_fn = jax.jit(prior_fn)

            elif prior_type == "ensemble_induced":
                rng_key0, _ = jax.random.split(rng_key)
                params_prior = self._pretraining_initialization(params)
                state_prior = state

                def prior_fn(inducing_inputs, model_params, rng_key, prior_cov, state):
                    return partial(
                        utils_linearization.induced_prior_fn_v0,
                        apply_fn=apply_fn,
                        state=state_prior,
                        rng_key=rng_key0,
                        task_id=task_id,
                        n_inducing_inputs=self.n_inducing_inputs,
                        architecture=self.architecture,
                        stochastic_linearization=self.stochastic_linearization_prior,
                        full_ntk=self.full_ntk,
                    )(inducing_inputs=inducing_inputs, params=params_prior)
                if jit_prior and not identity_cov:
                    prior_fn = jax.jit(prior_fn)

            else:
                if "fsvi" in self.model_type or "tdvi" in self.model_type:
                    raise ValueError("No prior type specified.")
        elif "mfvi" in self.model_type:
            prior_fn = lambda inducing_inputs, model_params, rng_key, prior_cov, state: [
                dtype_default(prior_mean),
                dtype_default(prior_cov)
            ]
        else:
            prior_fn = lambda inducing_inputs, model_params, rng_key, prior_cov, state: [
                dtype_default(prior_mean),
                dtype_default(prior_cov)
            ]

        return prior_fn

    def get_params_partition_fn(self, params):
        if "fsvi" in self.model_type or "tdvi" in self.model_type or "mfvi" in self.model_type:
            if self.final_layer_variational:
                variational_layers = list(params.keys())[-1:]  # TODO: set via input parameter
            else:
                variational_layers = list(params.keys())
        else:
            variational_layers = [""]

        def _get_params(params):
            variational_params, model_params = hk.data_structures.partition(lambda m, n, p: m in variational_layers, params)
            return variational_params, model_params

        return _get_params

    def get_trainable_params_fn(self, params):
        trainable_layers = list(params.keys())

        if self.final_layer_variational and self.fixed_inner_layers_variational_var:
            if self.start_var_opt == 0:
                get_trainable_params = lambda params: hk.data_structures.partition(
                    lambda m, n, p: (m in trainable_layers and 'mu' in n) or ('final' in m) or ('batchnorm' in m), params
                    # lambda m, n, p: (m in trainable_layers and 'mu' in n) or ('linear' in m), params  # use this if you want to make all linear layers stochastic
                )
            else:
                get_trainable_params = lambda params: hk.data_structures.partition(
                    lambda m, n, p: (m in trainable_layers and 'mu' in n) or ('batchnorm' in m), params
                )
        else:
            if self.start_var_opt == 0:
                if self.fixed_inner_layers_variational_var:
                    get_trainable_params = lambda params: hk.data_structures.partition(
                        lambda m, n, p: (m in trainable_layers and 'mu' in n) or ('batchnorm' in m), params
                    )
                else:
                    get_trainable_params = lambda params: hk.data_structures.partition(
                        lambda m, n, p: m in trainable_layers, params
                    )
            else:
                get_trainable_params = lambda params: hk.data_structures.partition(
                    lambda m, n, p: (m in trainable_layers and 'mu' in n) or ('batchnorm' in m), params
                )

        return get_trainable_params

    def kl_input_functions(
        self,
        apply_fn: Callable,
        predict_f: Callable,
        predict_f_deterministic: Callable,
        state: hk.State,
        params: hk.Params,
        prior_mean: str,
        prior_cov: str,
        rng_key,
        x_ood=None,
        y_ood=None,
        prior_type=None,
        task_id=None,
        jit_prior=True,
        identity_cov=False,
    ) -> Tuple[
        Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        Callable[[jnp.ndarray], List[jnp.ndarray]],
    ]:
        """
        @predict_f_deterministic: function to do forward pass
        @param prior_mean: example: "0.0"
        @param prior_cov: example: "0.0"
        @return:
            inducing_input_fn
            prior_fn: a function that takes in an array of inducing input points and return the mean
                and covariance of the outputs at those points
        """
        task_id = self.task_id if task_id is None else task_id
        prior_type = self.prior_type if prior_type is None else prior_type
        prior_mean = dtype_default(prior_mean)

        if self.prior_covs[0] != 0.0:
            prior_cov = []
            for i in range(len(self.prior_covs)):
                prior_cov.append(dtype_default(self.prior_covs[i]))
        else:
            prior_cov = dtype_default(prior_cov)

        inducing_input_fn = self.get_inducing_input_fn(x_ood=x_ood, y_ood=y_ood, rng_key=rng_key)
        prior_fn = self.get_prior_fn(
            apply_fn,
            predict_f,
            predict_f_deterministic,
            state,
            params,
            prior_mean,
            prior_cov,
            rng_key,
            prior_type,
            task_id,
            jit_prior,
            identity_cov,
        )

        return inducing_input_fn, prior_fn


def get_prior_params(
    params_init: hk.Params,
    prior_mean: str,
    prior_cov: str,
    batch_normalization: bool,
) -> hk.Params:
    prior_mean, prior_cov = dtype_default(prior_mean), dtype_default(prior_cov)

    params_mean = tree.map_structure(
        lambda p: jnp.ones_like(p) * prior_mean,
        hk.data_structures.filter(predicate_mean, params_init),
    )
    params_log_var = tree.map_structure(
        lambda p: jnp.ones_like(p) * jnp.log(prior_cov),
        hk.data_structures.filter(predicate_var, params_init),
    )

    params_prior = hk.data_structures.merge(params_mean, params_log_var)

    if batch_normalization:
        params_batchnorm = hk.data_structures.filter(predicate_batchnorm, params_init)
        params_prior = hk.data_structures.merge(params_prior, params_batchnorm)

    return params_prior


def piecewise_constant_schedule(init_value, boundaries, scale):
    """
    Return a function that takes in the update count and returns a step size.

    The step size is equal to init_value * (scale ** <number of boundaries points not greater than count>)
    """
    def schedule(count):
        v = init_value
        for threshold in boundaries:
            indicator = jnp.maximum(0.0, jnp.sign(threshold - count))
            v = v * indicator + (1 - indicator) * scale * v
        return v

    return schedule


def linear_schedule(init_value: float, scale:float, n_batches: int):
    """
    Return a function that takes in the update count and returns a step size.
    """
    def schedule(count):
        v = init_value
        i = count // n_batches + 1
        v = v / (scale * i)
        return v

    return schedule


def inv_sqrt_schedule(init_value: float, scale:float, n_batches: int):
    """
    Return a function that takes in the update count and returns a step size.
    """
    def schedule(count):
        v = init_value
        i = count // n_batches + 1
        v = v / (scale * (i ** 0.5))
        return v

    return schedule


def decide_prediction_type(data_training: str) -> str:
    if data_training in classification_datasets:
        prediction_type = "classification"
    elif data_training in regression_datasets:
        prediction_type = "regression"
    elif data_training == "online_rl":
        prediction_type = "regression"
    else:
        raise ValueError(f"Prediction type not recognized: {data_training}")
    return prediction_type
