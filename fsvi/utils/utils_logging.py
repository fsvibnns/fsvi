from functools import partial
import csv
import getpass
import os
import time
import pickle
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tabulate
import ast

from jax.tree_util import tree_flatten
from tensorflow_probability.substrates import jax as tfp

import uncertainty_metrics.numpy as um

tfd = tfp.distributions

from fsvi.utils import utils

import matplotlib.pyplot as plt

# import joypy
import seaborn as sns

sns.set()

import wandb

from IPython.display import set_matplotlib_formats

set_matplotlib_formats("pdf", "png")
plt.rcParams["savefig.dpi"] = 75
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = 6, 4
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["font.size"] = 16
plt.rcParams["lines.linewidth"] = 2.0
plt.rcParams["lines.markersize"] = 8
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["grid.linestyle"] = "-"
plt.rcParams["grid.linewidth"] = 1.0
plt.rcParams["legend.facecolor"] = "white"
# plt.rcParams['grid.color'] = "grey"
if getpass.getuser() == "ANON":
    plt.rcParams["text.usetex"] = True
    # plt.rcParams['font.family'] = "normal"
    # plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams[
        "text.latex.preamble"
    ] = "\\usepackage{subdepth} \\usepackage{amsfonts} \\usepackage{type1cm}"

columns_ood = [
    "ep",
    "tr_nll",
    "tr_acc",
    "te_nll",
    "te_acc",
    "te ece",
    "auc ent",
    "auc alea",
    "auc epis",
    "te_entr",
    "ood_entr",
    "elbo",
    "ellk",
    "kl",
    "time_ep",
    "time_eval",
]

columns_ood_large = [
    "ep",
    "tr_nll",
    "tr_acc",
    "te_nll",
    "te_acc",
    "ELBO",
    "fKL",
    "te_entr",
    "ood_entr",
    "noise_entr",
    "auc ent",
    "auc alea",
    "auc epis",
    "∇ logvar ll",
    "∇ mean ll",
    "∇ logvar fkl",
    "∇ mean fkl",
    "time_ep",
    "time_eval",
]

columns_ci = [
    "ep",
    "tr_nll",
    "tr_rmse",
    "te_nll",
    "te_rmse",
]

dtype_default = jnp.float32


class Logging:
    def __init__(
        self,
        task,
        model_type,
        architecture,
        model,
        apply_fn,
        metrics,
        loss,
        kl_evaluation,
        log_likelihood_evaluation,
        nll_grad_evaluation,
        task_evaluation,
        epoch_start,
        epochs,
        x_train_permuted,
        y_train_permuted,
        x_test,
        y_test,
        x_ood,
        n_train,
        val_frac,
        n_samples_eval,
        logging_frequency,
        debug,
        save,
        save_path,
        wandb_project,
        **kwargs,
    ):
        self.task = task
        self.model_type = model_type
        self.architecture = architecture
        self.model = model
        self.apply_fn = apply_fn
        self.metrics = metrics
        self.loss = loss
        self.kl_evaluation = kl_evaluation
        self.log_likelihood_evaluation = log_likelihood_evaluation
        self.nll_grad_evaluation = nll_grad_evaluation
        self.task_evaluation = task_evaluation
        self.epoch_start = epoch_start
        self.epochs = epochs
        self.kwargs = kwargs

        self.x_train_permuted = x_train_permuted
        self.y_train_permuted = y_train_permuted
        self.x_test = x_test
        self.y_test = y_test
        self.x_ood = x_ood

        self.n_train = n_train
        self.val_frac = val_frac
        self.n_samples_eval = n_samples_eval
        self.logging_frequency = logging_frequency
        self.debug = debug
        self.save = save
        self.save_path = save_path

        self.T, self.t0 = 0, 0
        self.feature_update = True

        self.res = {}

        self.res["epoch"] = []

        self.res["rmse_train"] = []
        self.res["rmse_test"] = []
        self.res["rmse_unnormalized_train"] = []
        self.res["rmse_unnormalized_test"] = []
        self.res["llk_train"] = []
        self.res["llk_test"] = []
        self.res["elbo"] = []
        self.res["log_likelihood"] = []
        self.res["kl"] = []

        self.res["accuracy_train"] = []
        self.res["accuracy_test"] = []
        self.res["accuracy_val"] = []

        self.res["entropy_test"] = []
        self.res["entropy_val"] = []
        self.res["entropy_ood"] = []
        self.res["entropy_ood_val"] = []

        self.res["expected_entropy_test"] = []
        self.res["expected_entropy_val"] = []
        self.res["expected_entropy_ood"] = []
        self.res["expected_entropy_ood_val"] = []

        self.res["auroc_entropy_ood"] = []
        self.res["auroc_entropy_ood_val"] = []
        self.res["auroc_expected_entropy_ood"] = []
        self.res["auroc_expected_entropy_ood_val"] = []
        self.res["auroc_pred_var_ood"] = []
        self.res["auroc_pred_var_ood_val"] = []

        self.res["pred_var_train"] = []
        self.res["var_params"] = []
        self.res["var_params_mean"] = []
        self.res["var_params_median"] = []
        self.res["var_params_max"] = []

        self.n_samples_train = 1

        if self.n_samples_eval == 0:
            if "map" not in self.model_type and "resnet" not in self.architecture:
                self.n_samples_eval = 10
            elif "map" not in self.model_type and "resnet" in self.architecture:
                self.n_samples_eval = 2
            else:
                if "dropout" in self.model_type:
                    self.n_samples_eval = 10
                else:
                    self.n_samples_eval = 1

        if "resnet" not in self.architecture:
            self.n_eval = 1000
        else:
            self.n_eval = 500

        self.n_batches_test = min(x_test.shape[0] // self.n_eval, 10)
        self.n_batches_ood = [min(x_ood.shape[0] // self.n_eval, 10) for x_ood in self.x_ood]

        self.ece_over_samples = False
        self.print = False
        self.save_wandb = (wandb_project != "not_specified")
        self.is_training = False

        # self.pred_fn_train = jax.jit(partial(self.model.predict_f_multisample, n_samples=self.n_samples_train, is_training=self.is_training))
        # self.pred_fn_eval = jax.jit(partial(self.model.predict_f_multisample, n_samples=self.n_samples_eval, is_training=self.is_training))
        self.pred_fn_train = self.model.predict_f_multisample
        self.pred_fn_eval = self.model.predict_f_multisample

        print(f"n_samples_eval: {self.n_samples_eval}")
        print(f"n_samples_train: {self.n_samples_train}")
        print(f"n_batches_test: {self.n_batches_test}")
        print(f"n_batches_ood: {self.n_batches_ood}")
        print(f"n_eval: {self.n_eval}")
        print(f"ece_over_samples: {self.ece_over_samples}")

    def training_progress(
        self,
        epoch,
        params,
        params_feature,
        state,
        rng_key,
        inducing_input_fn,
        prior_fn,
    ):
        t0 = time.time()
        if (epoch + 1) % 10 == 0 and "map" not in self.model_type:
            if "map" not in self.model_type and "resnet" not in self.architecture:
                n_samples_eval = 10
            elif "map" not in self.model_type and "resnet" in self.architecture:
                n_samples_eval = 10
        else:
            n_samples_eval = self.n_samples_eval

        permutation = np.random.permutation(self.n_train)[: self.n_eval*self.n_batches_test]

        predicted_f_samples_train_list = []
        predicted_f_mean_train_list = []
        predicted_f_var_train_list = []
        for i in range(self.n_batches_test):
            (
                predicted_f_samples_train,
                predicted_f_mean_train,
                predicted_f_var_train,
            ) = self.pred_fn_train(
                params,
                params_feature,
                state,
                self.x_train_permuted[permutation[i*self.n_eval:(i+1)*self.n_eval]],
                rng_key,
                self.n_samples_train,
                self.is_training
            )
            predicted_f_samples_train_list.append(predicted_f_samples_train)
            predicted_f_mean_train_list.append(predicted_f_mean_train)
            predicted_f_var_train_list.append(predicted_f_var_train)

        predicted_f_samples_train = np.concatenate(predicted_f_samples_train_list, axis=1)
        predicted_f_mean_train = np.concatenate(predicted_f_mean_train_list, axis=0)
        predicted_f_var_train = np.concatenate(predicted_f_var_train_list, axis=0)

        predicted_f_samples_test_list = []
        predicted_f_mean_test_list = []
        predicted_f_var_test_list = []
        for i in range(self.n_batches_test):
            x_test = self.x_test[i*self.n_eval:(i+1)*self.n_eval]
            # x_test_combined = np.concatenate([self.x_test[i*self.n_eval:(i+1)*self.n_eval], self.x_train_permuted[permutation][0:100]], axis=0)
            (
                predicted_f_samples_test,
                predicted_f_mean_test,
                predicted_f_var_test,
            ) = self.pred_fn_eval(
                params, params_feature, state, x_test, rng_key, n_samples_eval, self.is_training
            )
            predicted_f_samples_test_list.append(predicted_f_samples_test[:, 0:self.n_eval, :])
            predicted_f_mean_test_list.append(predicted_f_mean_test[0:self.n_eval, :])
            predicted_f_var_test_list.append(predicted_f_var_test[0:self.n_eval, :])

        predicted_f_samples_test = np.concatenate(predicted_f_samples_test_list, axis=1)
        predicted_f_mean_test = np.concatenate(predicted_f_mean_test_list, axis=0)
        predicted_f_var_test = np.concatenate(predicted_f_var_test_list, axis=0)

        predicted_f_samples_ood_list = []
        predicted_f_mean_ood_list = []
        predicted_f_var_ood_list = []
        for _ in range(len(self.x_ood)):
            predicted_f_samples_ood_list.append([])
            predicted_f_mean_ood_list.append([])
            predicted_f_var_ood_list.append([])

        loss = self.get_loss(
            params,
            params_feature,
            state,
            self.kwargs["kwargs"]["prior_mean"],
            self.kwargs["kwargs"]["prior_cov"],
            rng_key,
            permutation,
            inducing_input_fn,
            prior_fn,
        )

        for j, x_ood in enumerate(self.x_ood):
            for i in range(self.n_batches_ood[j]):
                _x_ood = x_ood[i*self.n_eval:(i+1)*self.n_eval]
                # _x_ood = np.concatenate([x_ood[i*self.n_eval:(i+1)*self.n_eval], self.x_train_permuted[permutation][0:100]], axis=0)
                (
                    predicted_f_samples_ood,
                    predicted_f_mean_ood,
                    predicted_f_var_ood,
                ) = self.pred_fn_eval(
                    params, params_feature, state, _x_ood, rng_key, n_samples_eval, self.is_training
                )
                predicted_f_samples_ood_list[j].append(predicted_f_samples_ood[:, 0:self.n_eval, :])
                predicted_f_mean_ood_list[j].append(predicted_f_mean_ood[0:self.n_eval, :])
                predicted_f_var_ood_list[j].append(predicted_f_var_ood[0:self.n_eval, :])

            predicted_f_samples_ood_list[j] = np.concatenate(predicted_f_samples_ood_list[j], axis=1)
            predicted_f_mean_ood_list[j] = np.concatenate(predicted_f_mean_ood_list[j], axis=0)
            predicted_f_var_ood_list[j] = np.concatenate(predicted_f_var_ood_list[j], axis=0)

        entropy_test = utils.predictive_entropy(
            jnp.mean(jax.nn.softmax(predicted_f_samples_test, axis=-1), axis=0)
        ).mean(0)

        accuracy_train = 100 * self.metrics.accuracy(
            jnp.mean(jax.nn.softmax(predicted_f_samples_train, axis=-1), axis=0), self.y_train_permuted[permutation]
        )
        accuracy_test = 100 * self.metrics.accuracy(
            jnp.mean(jax.nn.softmax(predicted_f_samples_test, axis=-1), axis=0), self.y_test[: self.n_eval * self.n_batches_test]
        )
        log_likelihood_train = jnp.sum(
            self.y_train_permuted[permutation]
            * jnp.mean(jax.nn.log_softmax(predicted_f_samples_train, axis=-1), axis=0)
        ) / (self.n_eval * self.n_batches_test)
        log_likelihood_test = jnp.sum(
            self.y_test[: self.n_eval * self.n_batches_test]
            * jnp.mean(jax.nn.log_softmax(predicted_f_samples_test, axis=-1), axis=0)
        ) / (self.n_eval * self.n_batches_test)

        auroc_entropy_list = []
        auroc_expected_entropy_list = []
        auroc_predictive_variance_list = []
        ood_entropy_list = []
        for j in range(len(self.x_ood)):
            entropy_ood = utils.predictive_entropy(
                jnp.mean(jax.nn.softmax(predicted_f_samples_ood_list[j], axis=-1), axis=0)
            ).mean(0)

            auroc_entropy = 100 * utils.auroc_logits(
                predicted_f_samples_test, predicted_f_samples_ood_list[j], score="entropy"
            )

            if "map" not in self.model_type or "dropout" in self.model_type:
                auroc_expected_entropy = 100 * utils.auroc_logits(
                    predicted_f_samples_test,
                    predicted_f_samples_ood_list[j],
                    score="expected entropy",
                )
                auroc_predictive_variance = 100 * utils.auroc_logits(
                    predicted_f_samples_test,
                    predicted_f_samples_ood_list[j],
                    score="mutual information",
                )

            else:
                auroc_expected_entropy = None
                auroc_predictive_variance = None

            auroc_entropy_list.append(auroc_entropy)
            auroc_expected_entropy_list.append(auroc_expected_entropy)
            auroc_predictive_variance_list.append(auroc_predictive_variance)
            ood_entropy_list.append(entropy_ood)

        if len(auroc_entropy_list) > 1:
            auroc_entropy_list_text = "| " + " ".join(format(x, "8.3f") for x in auroc_entropy_list) + "  |"
            auroc_expected_entropy_list_text = " ".join(format(x, "8.3f") for x in auroc_expected_entropy_list) + "  |" if auroc_expected_entropy_list[0] is not None else "----------"
            auroc_predictive_variance_list_text = " ".join(format(x, "8.3f") for x in auroc_predictive_variance_list) + "  |" if auroc_predictive_variance_list[0] is not None else "----------"
            ood_entropy_list_text = "| " + " ".join(format(x, "8.3f") for x in ood_entropy_list) + "  |"
        else:
            auroc_entropy_list_text = auroc_entropy_list[0]
            auroc_expected_entropy_list_text = auroc_expected_entropy_list[0]
            auroc_predictive_variance_list_text = auroc_predictive_variance_list[0]
            ood_entropy_list_text = ood_entropy_list[0]

        num_bins = 10
        labels_test = self.y_test[: self.n_eval * self.n_batches_test].argmax(-1)
        ece_test = um.ece(
            labels_test, jnp.mean(jax.nn.softmax(predicted_f_samples_test), axis=0), num_bins=num_bins
        )

        if self.ece_over_samples:
            ece_test_list = []
            for i in range(predicted_f_samples_test.shape[0]):
                ece_test_list.append(
                    um.ece(
                        labels_test,
                        jax.nn.softmax(predicted_f_samples_test[i]),
                        num_bins=num_bins,
                    )
                )
            ece_test = jnp.mean(jnp.array(ece_test_list))

        T = time.time()

        time_eval = T - t0
        time_ep = self.T - self.t0
        self.feature_update = True if accuracy_train > 98 else False

        # ## For debugging:
        # print('OOD:')
        # for i in range(20):
        #     print(jax.nn.softmax(predicted_f_mean_ood_list[j])[i].round(2))
        #
        # print('Test:')
        # for i in range(20):
        #     print(jax.nn.softmax(predicted_f_mean_test)[i].round(2))

        values = [
            epoch + 1,
            -log_likelihood_train,
            accuracy_train,
            -log_likelihood_test,
            accuracy_test,
            ece_test,
            auroc_entropy_list_text,
            auroc_expected_entropy_list_text,
            auroc_predictive_variance_list_text,
            entropy_test,
            ood_entropy_list_text,
            loss["elbo_train"],
            loss["log_likelihood_train"],
            loss["kl"],
            time_ep,
            time_eval,
        ]
        table = tabulate.tabulate([values], columns_ood, tablefmt="simple", floatfmt="8.3f")
        if epoch % 40 == 0:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]
        print(table)

        if self.save_wandb:
            self.save_results_to_wandb(
                epoch=epoch,
                loss=loss,
                log_likelihood_train=log_likelihood_train,
                log_likelihood_test=log_likelihood_test,
                accuracy_train=accuracy_train,
                accuracy_test=accuracy_test,
                entropy_test=entropy_test,
                ood_entropy_list=ood_entropy_list,
                auroc_entropy_list=auroc_entropy_list,
                auroc_expected_entropy_list=auroc_expected_entropy_list,
                auroc_predictive_variance_list=auroc_predictive_variance_list,
                ece_test=ece_test,
                time_ep=time_ep,
                time_eval=time_eval,
            )

        if self.save:
            self.save_parameters(epoch, params, state)
            self.save_results(
                epoch=epoch,
                log_likelihood_train=log_likelihood_train,
                log_likelihood_test=log_likelihood_test,
                accuracy_train=accuracy_train,
                accuracy_test=accuracy_test,
                entropy_test=entropy_test,
                entropy_ood=entropy_ood,
                auroc_entropy=auroc_entropy,
                auroc_expected_entropy=auroc_expected_entropy,
                auroc_predictive_variance=auroc_predictive_variance,
                ece_test=ece_test,
                time_ep=time_ep,
                time_eval=time_eval,
            )

    def training_progress_large(
        self,
        epoch,
        params,
        state,
        x_batch,
        y_batch,
        prior_mean,
        prior_cov,
        inducing_inputs,
        rng_key,
    ):
        t0 = time.time()

        permutation = np.random.permutation(self.n_train)[: self.n_eval]
        (
            predicted_f_samples_train,
            predicted_f_mean_train,
            _,
        ) = self.model.predict_f_multisample(
            params,
            state,
            self.x_train_permuted[permutation],
            rng_key,
            self.n_samples_eval,
            is_training = False,
        )
        (
            predicted_f_samples_test,
            predicted_f_mean_test,
            _,
        ) = self.model.predict_f_multisample(
            params, state, self.x_test, rng_key, self.n_samples_eval, is_training = False
        )
        (
            predicted_f_samples_ood,
            predicted_f_mean_ood,
            _,
        ) = self.model.predict_f_multisample(
            params, state, self.x_ood[: self.n_eval], rng_key, self.n_samples_eval, is_training = False
        )

        noise_image = jax.random.uniform(
            rng_key, list(self.x_test[: self.n_eval, :].shape), dtype_default, 0, 1
        )
        (
            predicted_f_samples_noise,
            predicted_f_mean_noise,
            _,
        ) = self.model.predict_f_multisample(
            params, state, noise_image, rng_key, self.n_samples_eval, is_training = False
        )

        entropy_test = utils.predictive_entropy(
            jax.nn.softmax(predicted_f_mean_test)
        ).mean(0)
        entropy_ood = utils.predictive_entropy(
            jax.nn.softmax(predicted_f_mean_ood)
        ).mean(0)
        entropy_noise = utils.predictive_entropy(
            jax.nn.softmax(predicted_f_mean_noise)
        ).mean(0)

        accuracy_train = 100 * self.metrics.accuracy(
            predicted_f_mean_train, self.y_train_permuted[permutation]
        )
        accuracy_test = 100 * self.metrics.accuracy(
            predicted_f_mean_test, self.y_test[: self.n_eval]
        )
        log_likelihood_train = (
            self.log_likelihood_evaluation(
                predicted_f_samples_train, self.y_train_permuted[permutation]
            )
            / self.n_eval
        )
        log_likelihood_test = (
            self.log_likelihood_evaluation(
                predicted_f_samples_test, self.y_test[: self.n_eval]
            )
            / self.n_eval
        )

        auroc_entropy = 100 * utils.auroc_logits(
            predicted_f_samples_test, predicted_f_samples_ood, score="entropy"
        )

        if "map" not in self.model_type:
            auroc_expected_entropy = 100 * utils.auroc_logits(
                predicted_f_samples_test,
                predicted_f_samples_ood,
                score="expected entropy",
            )
            auroc_predictive_variance = 100 * utils.auroc_logits(
                predicted_f_samples_test,
                predicted_f_samples_ood,
                score="mutual information",
            )
        else:
            auroc_expected_entropy = 0
            auroc_predictive_variance = 0

        if "fsvi" in self.model_type:
            elbo, _, kl, scale = self.metrics._elbo_fsvi_classification(
                params,
                state,
                prior_mean,
                prior_cov,
                x_batch,
                y_batch,
                inducing_inputs,
                rng_key,
                is_training = False,
            )
            grad_nll = jax.grad(self.nll_grad_evaluation, argnums=0)(
                params, state, x_batch, y_batch, rng_key, is_training = False,
            )
            grad_function_kl, kl_scale = jax.grad(self.kl_evaluation, argnums=0, has_aux=True)(
                params,
                state,
                prior_mean,
                prior_cov,
                x_batch,
                y_batch,
                inducing_inputs,
                rng_key,
            )

            predicate = (
                lambda module_name, name, value: name == "w_mu" or name == "b_mu"
            )
            grads_log_var_nll = tree_flatten(
                hk.data_structures.partition(predicate, grad_nll)[1]
            )[0]
            grads_mean_nll = tree_flatten(
                hk.data_structures.partition(predicate, grad_nll)[0]
            )[0]
            grads_log_var_function_kl = tree_flatten(
                hk.data_structures.partition(predicate, grad_function_kl)[1]
            )[0]
            grads_mean_function_kl = tree_flatten(
                hk.data_structures.partition(predicate, grad_function_kl)[0]
            )[0]

            grads_log_var_log_likelihood_mean = -np.mean(
                np.concatenate(
                    [
                        np.array(grads_log_var_nll[i].flatten())
                        for i in range(len(grads_log_var_nll))
                    ],
                    0,
                )
            )
            grads_mean_log_likelihood_mean = -np.mean(
                np.concatenate(
                    [
                        np.array(grads_mean_nll[i].flatten())
                        for i in range(len(grads_mean_nll))
                    ],
                    0,
                )
            )
            grads_log_var_function_kl_mean = np.mean(
                np.concatenate(
                    [
                        np.array(grads_log_var_function_kl[i].flatten())
                        for i in range(len(grads_log_var_function_kl))
                    ],
                    0,
                )
            )
            grads_mean_function_kl_mean = np.mean(
                np.concatenate(
                    [
                        np.array(grads_mean_function_kl[i].flatten())
                        for i in range(len(grads_mean_function_kl))
                    ],
                    0,
                )
            )
        else:
            elbo, kl, scale = 0, 0, 0
            grads_log_var_log_likelihood_mean = 0
            grads_mean_log_likelihood_mean = 0
            grads_log_var_function_kl_mean = 0
            grads_mean_function_kl_mean = 0

        fKL = kl * scale

        T = time.time()

        time_eval = T - t0
        time_ep = self.T - self.t0
        values = [
            epoch + 1,
            -log_likelihood_train,
            accuracy_train,
            -log_likelihood_test,
            accuracy_test,
            elbo,
            fKL,
            entropy_test,
            entropy_ood,
            entropy_noise,
            auroc_entropy,
            auroc_expected_entropy,
            auroc_predictive_variance,
            grads_log_var_log_likelihood_mean,
            grads_mean_log_likelihood_mean,
            grads_log_var_function_kl_mean,
            grads_mean_function_kl_mean,
            time_ep,
            time_eval,
        ]
        table = tabulate.tabulate(
            [values], columns_ood_large, tablefmt="simple", floatfmt="8.4f"
        )
        if epoch % 40 == 0:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]
        print(table)
        if self.save:
            self.save_parameters(epoch, params)
            self.save_results_large(
                epoch=epoch,
                log_likelihood_train=log_likelihood_train,
                log_likelihood_test=log_likelihood_test,
                accuracy_train=accuracy_train,
                accuracy_test=accuracy_test,
                elbo=elbo,
                fKL=fKL,
                entropy_test=entropy_test,
                entropy_ood=entropy_ood,
                entropy_noise=entropy_noise,
                auroc_entropy=auroc_entropy,
                auroc_expected_entropy=auroc_expected_entropy,
                auroc_predictive_variance=auroc_predictive_variance,
                grads_log_var_log_likelihood_mean=grads_log_var_log_likelihood_mean,
                grads_mean_log_likelihood_mean=grads_mean_log_likelihood_mean,
                grads_log_var_function_kl_mean=grads_log_var_function_kl_mean,
                grads_mean_function_kl_mean=grads_mean_function_kl_mean,
                time_ep=time_ep,
                time_eval=time_eval,
            )

    def training_progress_ci(
        self,
        epoch,
        params,
        state,
        rng_key,
    ):
        if (epoch + 1) % 10 == 0 and "map" not in self.model_type:
            self.n_samples_eval = 50
        else:
            self.n_samples_eval = 30

        (
            predicted_f_samples_train,
            predicted_f_mean_train,
            predicted_f_var_train,
        ) = self.model.predict_f_multisample(
            params,
            state,
            self.x_train_permuted,
            rng_key,
            self.n_samples_train,
            is_training = False,
        )
        (
            predicted_f_samples_test,
            predicted_f_mean_test,
            predicted_f_var_test,
        ) = self.model.predict_f_multisample(
            params, state, self.x_test, rng_key, self.n_samples_eval, is_training = False
        )

        rmse_train = jnp.sqrt(jnp.mean((predicted_f_mean_train - self.y_train_permuted) ** 2))
        rmse_test = jnp.sqrt(jnp.mean((predicted_f_mean_test - self.y_test) ** 2))
        log_likelihood_train = (
            self.log_likelihood_evaluation(
                predicted_f_samples_train, self.y_train_permuted
            )
            / self.n_eval
        )
        log_likelihood_test = (
            self.log_likelihood_evaluation(
                predicted_f_samples_test, self.y_test
            )
            / self.n_eval
        )

        values = [
            epoch + 1,
            -log_likelihood_train,
            rmse_train,
            -log_likelihood_test,
            rmse_test,
        ]
        table = tabulate.tabulate([values], columns_ci, tablefmt="simple", floatfmt="8.4f")
        if epoch % 40 == 0:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]
        if epoch < 10 or (epoch + 1) % self.logging_frequency == 0:
            print(table)

        # if self.save:
        #     self.save_parameters(epoch, params, state)

    #     def log_training_progress(
    #             self,
    #             epoch,
    #             params,
    #             state,
    #             x_batch,
    #             y_batch,
    #             x_test,
    #             y_test,
    #             x_ood,
    #             prior_mean,
    #             prior_cov,
    #             inducing_inputs,
    #             rng_key,
    #     ):
    #         permutation = np.random.permutation(self.n_train)[:self.n_eval]

    #         _, predicted_y_mean_train, _ = self.model.predict_y_multisample(params, state, self.x_train_permuted[permutation], rng_key, self.n_samples_eval)
    #         predicted_y_samples_test, predicted_y_mean_test, predicted_y_var_test = self.model.predict_y_multisample(params, state, x_test, rng_key, self.n_samples_eval)
    #         predicted_y_samples_ood, predicted_y_mean_ood, predicted_y_var_ood = self.model.predict_y_multisample(params, state, x_ood, rng_key, self.n_samples_eval)

    #         accuracy_train = self.metrics.accuracy(predicted_y_mean_train, self.y_train_permuted[permutation])
    #         accuracy_test = self.metrics.accuracy(predicted_y_mean_test, self.y_test)

    #         auroc_entropy = utils.auroc(predicted_y_samples_test, predicted_y_samples_ood, score='entropy')
    #         auroc_expected_entropy = utils.auroc(predicted_y_samples_test, predicted_y_samples_ood, score='expected entropy')
    #         auroc_predictive_variance = utils.auroc(predicted_y_samples_test, predicted_y_samples_ood, score='mutual information')

    #         if self.print:
    #             print(f'\n--- Epoch {epoch + 1} ---\nAccuracy (Train): {accuracy_train*100:2.2f}%  |  Accuracy (Test): {accuracy_test*100:2.2f}%\nAUROC (Entropy):  {auroc_entropy*100:2.2f}%  |  AUROC (Expected Entropy): {auroc_expected_entropy*100:2.2f}%  |  AUROC (Predictive Variance): {auroc_predictive_variance*100:2.2f}%')

    #         if 'fsvi' in self.model_type:
    #             elbo, log_likelihood, kl, scale = self.metrics._elbo_fsvi_classification(
    #                 params, state, prior_mean, prior_cov, x_batch, y_batch, inducing_inputs, rng_key
    #             )

    #             if self.print:
    #                 print(f'\nELBO:     {elbo:7.0f}  |  Log-Likelihood:{log_likelihood:7.0f}')
    #                 print(f'KL:       {kl:7.0f}  |  scale * KL:    {scale * kl:7.0f}')

    #             if self.save:
    #                 grad_nll = jax.grad(self.nll_grad_evaluation, argnums = 0)(params, state, x_batch, y_batch, rng_key)
    #                 grad_function_kl = jax.grad(self.kl_evaluation, argnums = 0)(params, state, prior_mean, prior_cov, x_batch, y_batch, inducing_inputs, rng_key)

    #                 predicate = lambda module_name, name, value: name == 'w_mu' or name == 'b_mu'
    #                 grads_log_var_nll = tree_flatten(hk.data_structures.partition(predicate, grad_nll)[1])[0]
    #                 grads_mean_nll = tree_flatten(hk.data_structures.partition(predicate, grad_nll)[0])[0]
    #                 grads_log_var_function_kl = tree_flatten(hk.data_structures.partition(predicate, grad_function_kl)[1])[0]
    #                 grads_mean_function_kl = tree_flatten(hk.data_structures.partition(predicate, grad_function_kl)[0])[0]

    #                 grads_log_var_log_likelihood_mean = -np.mean(np.concatenate([np.array(grads_log_var_nll[i].flatten()) for i in range(len(grads_log_var_nll))], 0))
    #                 grads_mean_log_likelihood_mean = -np.mean(np.concatenate([np.array(grads_mean_nll[i].flatten()) for i in range(len(grads_mean_nll))], 0))
    #                 grads_log_var_function_kl_mean = np.mean(np.concatenate([np.array(grads_log_var_function_kl[i].flatten()) for i in range(len(grads_log_var_function_kl))], 0))
    #                 grads_mean_function_kl_mean = np.mean(np.concatenate([np.array(grads_mean_function_kl[i].flatten()) for i in range(len(grads_mean_function_kl))], 0))

    #                 if self.print:
    #                     print(f'\ngrad-log-like-log-var:     {grads_log_var_log_likelihood_mean:.5f}  |  grad-log-like-mean:     {grads_mean_log_likelihood_mean:.5f}')
    #                     print(f'grad-function-kl-log-var:  {grads_log_var_function_kl_mean:.5f}  |  grad-function-kl-mean:  {grads_mean_function_kl_mean:.5f}')

    #                 file_name = f'{self.save_path}/metrics_all_iterations.csv'
    #                 with open(file_name, 'a') as metrics_file:
    #                     metrics_header = [
    #                         'Epoch',
    #                         'Train Loss',
    #                         'Train Log-Likelihood',
    #                         'KL',
    #                         'Train Accuracy',
    #                         'Test Accuracy',
    #                         'OOD AUROC (Entropy)',
    #                         'OOD AUROC (Expected Entropy)',
    #                         'OOD AUROC (Predictive Variance)',
    #                         'Grad-Log-Like-Log-Var',
    #                         'Grad-Log-Like-Mean',
    #                         'Grad-Function-KL-Log-Var',
    #                         'Grad-Function-KL-Mean',
    #                     ]
    #                     writer = csv.DictWriter(metrics_file, fieldnames=metrics_header)
    #                     if os.stat(file_name).st_size == 0:
    #                         writer.writeheader()
    #                     writer.writerow({
    #                         'Epoch': epoch+1,
    #                         'Train Loss': elbo,
    #                         'Train Log-Likelihood': log_likelihood,
    #                         'KL': scale*kl,
    #                         'Train Accuracy': accuracy_train,
    #                         'Test Accuracy': accuracy_test,
    #                         'OOD AUROC (Entropy)': auroc_entropy,
    #                         'OOD AUROC (Expected Entropy)': auroc_expected_entropy,
    #                         'OOD AUROC (Predictive Variance)': auroc_predictive_variance,
    #                         'Grad-Log-Like-Log-Var': grads_log_var_log_likelihood_mean,
    #                         'Grad-Log-Like-Mean': grads_mean_log_likelihood_mean,
    #                         'Grad-Function-KL-Log-Var': grads_log_var_function_kl_mean,
    #                         'Grad-Function-KL-Mean': grads_mean_function_kl_mean,
    #                     })
    #                     metrics_file.close()

    #     def log_training_metrics(
    #             self,
    #             epoch,
    #             x_batch,
    #             y_batch,
    #             x_test,
    #             y_test,
    #             x_ood,
    #             y_ood,
    #             inducing_inputs,
    #             params,
    #             state,
    #             prior_mean,
    #             prior_cov,
    #             rng_key,
    #     ):
    #         if (epoch + 1) % self.logging_frequency == 0 or epoch + 1 == self.epochs:
    #             rng_key, subkey = jax.random.split(rng_key)

    #             n_samples = 50

    #             permutation = np.random.permutation(self.n_train)[:self.n_eval]
    #             _, predicted_y_mean_train, _ = self.model.predict_y_multisample(params, state, self.x_train_permuted[permutation], rng_key, n_samples)
    #             predicted_y_samples_test, predicted_y_mean_test, predicted_y_var_test = self.model.predict_y_multisample(params, state, self.x_test, rng_key, n_samples)
    #             predicted_y_samples_ood, predicted_y_mean_ood, predicted_y_var_ood = self.model.predict_y_multisample(params, state, self.x_ood[:self.n_eval], rng_key, n_samples)
    #             noise_image = jax.random.uniform(rng_key, list(x_test[:self.n_eval, :].shape), dtype_default, 0, 1)
    #             predicted_y_samples_noise, predicted_y_mean_noise, predicted_y_var_noise = self.model.predict_y_multisample(params, state, noise_image, rng_key, n_samples)

    #             accuracy_train = self.metrics.accuracy(predicted_y_mean_train, self.y_train_permuted[permutation])
    #             accuracy_test = self.metrics.accuracy(predicted_y_mean_test, self.y_test)

    #             auroc_entropy_ood = utils.auroc(predicted_y_samples_test, predicted_y_samples_ood, score='entropy')
    #             auroc_expected_entropy_ood = utils.auroc(predicted_y_samples_test, predicted_y_samples_ood, score='expected entropy')
    #             auroc_mutual_information_ood = utils.auroc(predicted_y_samples_test, predicted_y_samples_ood, score='mutual information')

    #             entropy_test = utils.predictive_entropy(predicted_y_samples_test.mean(0)).mean(0)
    #             entropy_ood = utils.predictive_entropy(predicted_y_samples_ood.mean(0)).mean(0)
    #             entropy_noise = utils.predictive_entropy(predicted_y_samples_noise.mean(0)).mean(0)

    #             expected_entropy_test = utils.predictive_entropy(predicted_y_samples_test).mean(0).mean(0)
    #             expected_entropy_ood = utils.predictive_entropy(predicted_y_samples_ood).mean(0).mean(0)
    #             expected_entropy_noise = utils.predictive_entropy(predicted_y_samples_noise).mean(0).mean(0)

    #             labels_test = y_test[:self.n_eval].argmax(-1)
    #             labels_ood = y_ood[:self.n_eval].argmax(-1)
    #             labels_noise = jax.random.randint(rng_key, shape=[self.n_eval], minval=0, maxval=10)

    #             num_bins = 10
    #             ece_test = um.ece(labels_test, predicted_y_mean_test, num_bins=num_bins)
    #             ece_ood = um.ece(labels_ood, predicted_y_mean_ood, num_bins=num_bins)
    #             ece_noise = um.ece(labels_noise, predicted_y_mean_noise, num_bins=num_bins)

    #             if "fsvi" in self.model_type:
    #                 elbo, log_likelihood, kl, scale = self.metrics._elbo_fsvi_classification(
    #                     params, state, prior_mean,
    #                     prior_cov, x_batch, y_batch,
    #                     inducing_inputs, rng_key
    #                 )
    #                 loss_train = -elbo

    #                 self.res['elbo'].append(elbo)
    #                 self.res['log_likelihood'].append(log_likelihood)
    #                 self.res['kl'].append(kl)

    #                 predicate = lambda module_name, name, value: name == 'w_mu' or name == 'b_mu'
    #                 params_log_var = tree_flatten(hk.data_structures.partition(predicate, params)[1])[0]
    #                 self.res['var_params'].append(np.exp(np.concatenate([np.array(params_log_var[i].flatten()) for i in range(len(params_log_var))], 0)))
    #                 self.res['var_params_mean'].append(np.mean(np.exp(np.concatenate([np.array(params_log_var[i].flatten()) for i in range(len(params_log_var))], 0))))
    #                 self.res['var_params_median'].append(np.median(np.exp(np.concatenate([np.array(params_log_var[i].flatten()) for i in range(len(params_log_var))], 0))))
    #                 self.res['var_params_max'].append(np.max(np.exp(np.concatenate([np.array(params_log_var[i].flatten()) for i in range(len(params_log_var))], 0))))
    #             else:
    #                 if "fsvi" in self.model_type:
    #                     elbo, log_likelihood, kl, scale = self.metrics._elbo_mfvi_classification(
    #                         params, state, prior_mean,
    #                         prior_cov, x_batch, y_batch,
    #                         inducing_inputs, rng_key
    #                     )
    #                     loss_train = -elbo
    #                 else:
    #                     elbo, log_likelihood, kl, scale = 0, 0, 0, 0
    #                     loss_train = self.loss(params, state, prior_mean, prior_cov, x_batch, y_batch, inducing_inputs, rng_key)
    #                     loss_test = self.loss(params, state, prior_mean, prior_cov, x_test, y_test, inducing_inputs, rng_key)

    #                 self.res['elbo'].append(elbo)
    #                 self.res['log_likelihood'].append(log_likelihood)
    #                 self.res['kl'].append(kl)
    #                 self.res['var_params'].append(0)
    #                 self.res['var_params_mean'].append(0)
    #                 self.res['var_params_median'].append(0)
    #                 self.res['var_params_max'].append(0)

    #             self.res['epoch'].append(epoch + 1 + self.epoch_start)

    #             self.res['accuracy_train'].append(accuracy_train)
    #             self.res['accuracy_test'].append(accuracy_test)

    #             self.res['entropy_test'].append(entropy_test)
    #             self.res['entropy_ood'].append(entropy_ood)

    #             self.res['expected_entropy_test'].append(expected_entropy_test)
    #             self.res['expected_entropy_ood'].append(expected_entropy_ood)

    #             self.res['auroc_entropy_ood'].append(auroc_entropy_ood*100)
    #             self.res['auroc_expected_entropy_ood'].append(auroc_expected_entropy_ood*100)
    #             self.res['auroc_pred_var_ood'].append(auroc_mutual_information_ood*100)

    #             print(f'\n---Epoch {epoch + 1 + self.epoch_start}---')

    #             print(f'Loss (Train): {loss_train:7.0f}  |  Accuracy (Train):       {accuracy_train*100:2.2f}%  |  Accuracy (Test):       {accuracy_test*100:2.2f}%')

    #             if 'fsvi' in self.model_type or 'mfvi' in self.model_type:
    #                 print(f'\nELBO:         {elbo:7.0f}  |  Log-Likelihood:        {log_likelihood:7.0f}')
    #                 print(f'KL:           {kl:7.0f}  |  scale * KL:            {scale * kl:7.0f}')

    #             print(f'\nAUROC (Entropy):       {auroc_entropy_ood*100:2.2f}%  |  AUROC (Expected Entropy):       {auroc_expected_entropy_ood*100:2.2f}%  |  AUROC (MI):       {auroc_mutual_information_ood*100:2.2f}%')
    #             if self.val_frac > 0:
    #                 print(f'AUROC (Entropy) (Val): {auroc_entropy_ood*100:2.2f}%  |  AUROC (Expected Entropy) (Val): {auroc_expected_entropy_ood*100:2.2f}%  |  AUROC (MI) (Val): {auroc_mutual_information_ood*100:2.2f}%')

    #             print(f'\nEntropy (Test):          {entropy_test:1.2f}  |  Expected Entropy (Test):          {expected_entropy_test:1.2f}  |  ECE (Test):         {ece_test:1.2f}')
    #             print(f'Entropy (OOD):           {entropy_ood:1.2f}  |  Expected Entropy (OOD):           {expected_entropy_ood:1.2f}  |  ECE (OOD):          {ece_ood:1.2f}')
    #             print(f'Entropy (Noise):         {entropy_noise:1.2f}  |  Expected Entropy (Noise):         {expected_entropy_noise:1.2f}  |  ECE (Noise):        {ece_noise:1.2f}')

    #             if self.save:
    #                 file_name = f'{self.save_path}/metrics.csv'
    #                 with open(file_name, 'a') as metrics_file:
    #                     metrics_header = [
    #                         'Epoch',
    #                         'Train Loss',
    #                         'Train Log-Likelihood',
    #                         'KL',
    #                         'Train Accuracy',
    #                         'Test Accuracy',
    #                         'OOD AUROC (Predictive Entropy)',
    #                         'OOD AUROC (Expected Predictive Entropy)',
    #                         'OOD AUROC (Predictive Variance)',
    #                         'Test Predictive Entropy',
    #                         'OOD Predictive Entropy',
    #                         'Noise Predictive Entropy',
    #                         'Test ECE',
    #                         'OOD ECE',
    #                         'Noise ECE',
    #                         'Variance Parameter Mean',
    #                         'Variance Parameter Median',
    #                         'Variance Parameter Max',
    #                     ]
    #                     writer = csv.DictWriter(metrics_file, fieldnames=metrics_header)
    #                     if os.stat(file_name).st_size == 0:
    #                         writer.writeheader()
    #                     writer.writerow({
    #                         'Epoch': epoch + 1 + self.epoch_start,
    #                         'Train Loss': loss_train,
    #                         'Train Log-Likelihood': log_likelihood,
    #                         'KL': scale*kl,
    #                         'Train Accuracy': accuracy_train,
    #                         'Test Accuracy': accuracy_test,
    #                         'OOD AUROC (Expected Predictive Entropy)': auroc_expected_entropy_ood,
    #                         'OOD AUROC (Predictive Entropy)': auroc_entropy_ood,
    #                         'OOD AUROC (Predictive Variance)': auroc_mutual_information_ood,
    #                         'Test Predictive Entropy': entropy_test,
    #                         'OOD Predictive Entropy': entropy_ood,
    #                         'Noise Predictive Entropy': entropy_noise,
    #                         'Test ECE': ece_test,
    #                         'OOD ECE': ece_ood,
    #                         'Noise ECE': ece_noise,
    #                         'Variance Parameter Mean': self.res['var_params_mean'][-1],
    #                         'Variance Parameter Median': self.res['var_params_median'][-1],
    #                         'Variance Parameter Max': self.res['var_params_max'][-1],
    #                     })
    #                     metrics_file.close()

    #         ####### Saving final outputs #######
    #         if self.save and epoch + 1 == self.epochs:
    #             np.save(f'{self.save_path}/predicted_labels_test.npy', predicted_y_mean_test)
    #             np.save(f'{self.save_path}/predicted_labels_ood.npy', predicted_y_mean_ood)
    #             np.save(f'{self.save_path}/predicted_labels_noise.npy', predicted_y_mean_noise)

    def plot_1d(
        self,
        epoch,
        params,
        state,
        rng_key,
        x,
        x_train,
        x_test_features,
        y_train,
        inducing_inputs,
        prior_mean,
        prior_cov,
        noise_var,
        y_std,
        plot_progress,
        prior_fn,
    ):
        if epoch % int((self.logging_frequency / 10)) == 0:
            (
                preds_f_samples_train,
                preds_f_mean_train,
                preds_f_var_train,
            ) = self.model.predict_f_multisample(
                params, params, state, x_train, rng_key, self.n_samples_eval, is_training = False,
            )
            self.res["epoch"].append(epoch + 1)

            m, v = preds_f_mean_train, preds_f_var_train
            self.res["pred_var_train"].append(v.mean())

            y_train_tiled = jnp.tile(
                y_train, (preds_f_samples_train.shape[0], 1, 1)
            ).reshape(preds_f_samples_train.shape[0], -1)[:, :, None]
            d = y_train_tiled - preds_f_samples_train
            self.res["rmse_train"].append(np.mean(np.mean(d ** 2, 1) ** 0.5))

            if "fsvi" in self.model_type:
                _prior_mean, _prior_cov = prior_fn(
                    inducing_inputs=inducing_inputs,
                    model_params=None,  # TODO: Fix this in case model_params are needed
                    rng_key=rng_key,
                    prior_cov=dtype_default(prior_cov),
                    state=state,
                )
                elbo, log_likelihood, kl, scale = self.metrics._elbo_fsvi_regression(
                    params,
                    params,
                    state,
                    _prior_mean,
                    _prior_cov,
                    x_train,
                    y_train,
                    inducing_inputs,
                    rng_key,
                    is_training = False,
                )

                self.res["elbo"].append(elbo)
                self.res["log_likelihood"].append(log_likelihood)
                self.res["kl"].append(kl)

                predicate = (
                    lambda module_name, name, value: name == "w_mu" or name == "b_mu"
                )
                _, params_log_var = hk.data_structures.partition(predicate, params)
            elif "tdvi" in self.model_type:
                elbo, log_likelihood, kl, scale = self.metrics._elbo_tdvi_regression(
                    params,
                    state,
                    prior_mean,
                    prior_cov,
                    x_train,
                    y_train,
                    inducing_inputs,
                    rng_key,
                    is_training = False,
                )
                self.res["elbo"].append(elbo)
                self.res["log_likelihood"].append(log_likelihood)
                self.res["kl"].append(kl)

                predicate = (
                    lambda module_name, name, value: name == "w_mu" or name == "b_mu"
                )
                _, params_log_var = hk.data_structures.partition(predicate, params)
            else:
                self.res["elbo"].append(0)
                self.res["log_likelihood"].append(0)
                self.res["kl"].append(0)

        if epoch % self.logging_frequency == 0 and epoch > 100:
            print(
                f"\nEpoch: {epoch}\nTrain RMSE: {self.res['rmse_train'][-1]:.2f}  |  Train Log-Likelihood: {self.res['log_likelihood'][-1]:.2f}"
            )
            # print(
            #     f"ELBO: {self.res['elbo'][-1]:.0f}  |  Log-Likelihood: {self.res['log_likelihood'][-1]:.0f}  |  KL: {self.res['kl'][-1]:.0f}"
            # )

            (
                predicted_label_samples_test,
                predicted_labels_mean_test,
                predicted_labels_var_test,
            ) = self.model.predict_f_multisample(
                params, params, state, x_test_features, rng_key, self.n_samples_eval, is_training = False
            )
            y_test_samples = predicted_label_samples_test[:, :, 0]
            y_test_mean = predicted_labels_mean_test[:, 0]
            y_test_var = predicted_labels_var_test[:, 0]

            if "snelson" in self.task:
                xlim = (-6, 6)
                ylim = (-2.5, 2.5)
            elif "oat1d" in self.task:
                xlim = (-6, 6)
                ylim = (-2.5, 2.5)
            elif "solar" in self.task:
                xlim = (-0.25, 1.25)
                ylim = (-2, 2.5)
            elif "subspace_inference" in self.task:
                # xlim = (-3, 3)
                # ylim = None
                # ylim = (-10, 10)
                xlim = (-3, 3)
                ylim = (-2.5, 2.5)
            else:
                ValueError("No valid dataset specified.")

            plt.figure(figsize=(6, 4))
            plt.plot(
                self.x_test, y_test_mean, color="k", label="Predictive Mean", zorder=5
            )
            for i in range(self.n_samples_eval):
                if i == 0:
                    plt.plot(
                        self.x_test,
                        y_test_samples[i : i + 1, :].T,
                        linewidth=0.5,
                        color="xkcd:blue",
                        label="Function Draw",
                        zorder=3,
                        alpha=0.3,
                    )
                else:
                    plt.plot(
                        self.x_test,
                        y_test_samples[i : i + 1, :].T,
                        linewidth=0.5,
                        color="xkcd:blue",
                        zorder=3,
                        alpha=0.3,
                    )
            plt.fill_between(
                jnp.squeeze(self.x_test),
                jnp.squeeze(y_test_mean) - np.sqrt(noise_var + y_test_var),
                jnp.squeeze(y_test_mean) + np.sqrt(noise_var + y_test_var),
                color="C0",
                alpha=0.2,
            )
            plt.fill_between(
                jnp.squeeze(self.x_test),
                jnp.squeeze(y_test_mean) - 2 * np.sqrt(noise_var + y_test_var),
                jnp.squeeze(y_test_mean) + 2 * np.sqrt(noise_var + y_test_var),
                color="C0",
                alpha=0.2,
            )
            plt.scatter(x, y_train, s=10, color="r", label="Training Data", zorder=2)
            if xlim is not None:
                plt.xlim(xlim)
            if ylim is not None:
                plt.ylim(ylim)
            plt.legend()
            plt.grid(True)
            plt.savefig(
                f"figures/snelson/snelson_fsvi_predictive_distribution__samples_{self.n_samples_eval}.pdf",
                bbox_inches="tight",
            )
            plt.show()

            if plot_progress:
                fig, axs = plt.subplots(3, 2, figsize=(12, 12))

                axs[0, 0].plot(self.res["epoch"], self.res["rmse_train"])
                axs[0, 0].set_title("Standardized RMSE")

                axs[0, 1].plot(self.res["epoch"], self.res["elbo"])
                axs[0, 1].set_title(f"ELBO: {self.res['elbo'][-1]:.0f}")

                axs[1, 1].plot(self.res["epoch"], self.res["log_likelihood"])
                axs[1, 1].set_title(
                    f"(Expected) Log-Likelihood: {self.res['log_likelihood'][-1]:.0f}"
                )

                axs[2, 1].plot(self.res["epoch"], self.res["kl"])
                axs[2, 1].set_title(f"KL: {self.res['kl'][-1]:.0f}")

                plt.show()

            zoom = False
            if zoom:
                plot_range = int(epoch * 0.8 / self.logging_frequency)

                fig, axs = plt.subplots(3, 2, figsize=(12, 12))

                axs[0, 0].plot(
                    self.res["epoch"][-plot_range:],
                    self.res["rmse_train"][-plot_range:],
                )
                axs[0, 0].set_title("Standardized RMSE")

                axs[0, 1].plot(
                    self.res["epoch"][-plot_range:], self.res["elbo"][-plot_range:]
                )
                axs[0, 1].set_title(f"ELBO: {self.res['elbo'][-1]:.0f}")

                axs[1, 1].plot(
                    self.res["epoch"][-plot_range:],
                    self.res["log_likelihood"][-plot_range:],
                )
                axs[1, 1].set_title(
                    f"(Expected) Log-Likelihood: {self.res['log_likelihood'][-1]:.0f}"
                )

                axs[2, 1].plot(
                    self.res["epoch"][-plot_range:], self.res["kl"][-plot_range:]
                )
                axs[2, 1].set_title(f"KL: {self.res['kl'][-1]:.0f}")

                plt.show()

    def print_regression(
        self,
        epoch,
        params,
        state,
        rng_key,
        x,
        x_train,
        x_test_features,
        y_train,
        inducing_inputs,
        prior_mean,
        prior_cov,
        noise_var,
        y_std,
        plot_progress,
    ):
        noise_std = noise_var ** 0.5
        y_std = dtype_default(y_std)

        if epoch % int((self.logging_frequency / 10)) == 0:
            self.res["epoch"].append(epoch + 1)

            (
                preds_f_samples_train,
                preds_f_mean_train,
                preds_f_var_train,
            ) = self.model.predict_f_multisample(
                params, state, x_train, rng_key, self.n_samples_eval, is_training = False
            )

            (
                preds_f_samples_test,
                preds_f_mean_test,
                preds_f_var_test,
            ) = self.model.predict_f_multisample(
                params, state, self.x_test, rng_key, self.n_samples_eval, is_training = False
            )

            y_train_tiled = jnp.tile(
                y_train, (preds_f_samples_train.shape[0], 1, 1)
            ).reshape(preds_f_samples_train.shape[0], -1)[:, :, None]
            y_test_tiled = jnp.tile(
                self.y_test, (preds_f_samples_test.shape[0], 1, 1)
            ).reshape(preds_f_samples_test.shape[0], -1)[:, :, None]


            d_train_tiled = y_train_tiled - preds_f_samples_train
            du_train_tiled = d_train_tiled * y_std
            # d_train = y_train - preds_f_mean_train
            # du_train = d_train * y_std

            d_test_tiled = y_test_tiled - preds_f_samples_test
            du_test_tiled = d_test_tiled * y_std
            # d_test = y_test - preds_f_mean_test
            # du_test = d_test * y_std

            rmse_train = np.mean(np.mean(d_train_tiled ** 2, 1) ** 0.5)
            rmse_unnormalized_train = np.mean(np.mean(du_train_tiled ** 2, 1) ** 0.5)
            likelihood_fn_train = tfd.Normal(preds_f_samples_train * y_std, noise_std * y_std)
            llk_train = dtype_default(jnp.mean(jnp.mean(likelihood_fn_train.log_prob(y_train_tiled * y_std), 0), 0)[0])

            rmse_test = np.mean(np.mean(d_test_tiled ** 2, 1) ** 0.5)
            rmse_unnormalized_test = np.mean(np.mean(du_test_tiled ** 2, 1) ** 0.5)
            likelihood_fn_test = tfd.Normal(preds_f_samples_test * y_std, noise_std * y_std)
            llk_test = dtype_default(jnp.mean(jnp.mean(likelihood_fn_test.log_prob(y_test_tiled * y_std), 0), 0)[0])

            self.res["llk_train"].append(llk_train)
            self.res["rmse_train"].append(rmse_train)
            self.res["rmse_unnormalized_train"].append(rmse_unnormalized_train)

            self.res['llk_test'].append(llk_test)
            self.res['rmse_test'].append(rmse_test)
            self.res['rmse_unnormalized_test'].append(rmse_unnormalized_test)

            # m_train, v_train = preds_f_mean_train, preds_f_var_train
            # self.res["pred_var_train"].append(v_train.mean())
            # m_test, v_test = preds_f_mean_test, preds_f_var_test
            # self.res['pred_var_test'].append(v_test.mean())

            if "fsvi" in self.model_type:
                elbo, log_likelihood, kl, scale = self.metrics._elbo_fsvi_regression(
                    params,
                    state,
                    prior_mean,
                    prior_cov,
                    x_train,
                    y_train,
                    inducing_inputs,
                    rng_key,
                    is_training = False,
                )
                self.res["elbo"].append(elbo)
                self.res["log_likelihood"].append(log_likelihood)
                self.res["kl"].append(kl)

                predicate = (
                    lambda module_name, name, value: name == "w_mu" or name == "b_mu"
                )
                _, params_log_var = hk.data_structures.partition(predicate, params)
            if "tdvi" in self.model_type:
                elbo, log_likelihood, kl, scale = self.metrics._elbo_tdvi_regression(
                    params,
                    state,
                    prior_mean,
                    prior_cov,
                    x_train,
                    y_train,
                    inducing_inputs,
                    rng_key,
                    is_training = False,
                )
                self.res["elbo"].append(elbo)
                self.res["log_likelihood"].append(log_likelihood)
                self.res["kl"].append(kl)

                predicate = (
                    lambda module_name, name, value: name == "w_mu" or name == "b_mu"
                )
                _, params_log_var = hk.data_structures.partition(predicate, params)
            else:
                self.res["elbo"].append(0)
                self.res["log_likelihood"].append(0)
                self.res["kl"].append(0)

        if epoch % self.logging_frequency == 0 and epoch > 100:
            print(
                f"epoch: {epoch}  |  rmse_tr: {self.res['rmse_unnormalized_train'][-1]:.2f}  |  llk_tr: {self.res['llk_train'][-1]:.2f}  |  "
                f"rmse_te: {self.res['rmse_unnormalized_test'][-1]:.2f}  |  llk_te: {self.res['llk_test'][-1]:.2f}  |  "
                f"elbo: {self.res['elbo'][-1]:.0f}  |  ellk: {self.res['log_likelihood'][-1]:.0f}  |  kl: {self.res['kl'][-1]:.0f}"
            )

    def get_loss(
            self,
            params,
            params_feature,
            state,
            prior_mean,
            prior_cov,
            rng_key,
            permutation,
            inducing_input_fn,
            prior_fn,
    ):
        loss = {}
        elbos_train, log_likelihoods_train = [], []
        elbos_test, log_likelihoods_test = [], []
        kls = []
        for i in range(self.n_batches_test):
            inputs_train = self.x_train_permuted[permutation[i*self.n_eval:(i+1)*self.n_eval]]
            targets_train = self.y_train_permuted[permutation[i*self.n_eval:(i+1)*self.n_eval]]
            inputs_test = self.x_test[i*self.n_eval:(i+1)*self.n_eval]
            targets_test = self.y_test[i*self.n_eval:(i+1)*self.n_eval]

            if "fsvi" in self.model_type:
                inducing_inputs = inducing_input_fn(inputs_train, rng_key)
                _prior_mean, _prior_cov = prior_fn(
                    inducing_inputs=inducing_inputs,
                    model_params=None,  # TODO: Fix this in case model_params are needed
                    rng_key=rng_key,
                    prior_cov=dtype_default(prior_cov),
                    state=state,
                )

                # _prior_cov = jnp.ones_like(prior_cov) * 1e-4

                _, log_likelihood_train, kl, scale = self.metrics._elbo_fsvi_classification(
                    params,
                    params_feature,
                    state,
                    _prior_mean,
                    _prior_cov,
                    inputs_train,
                    targets_train,
                    inducing_inputs,
                    rng_key,
                    is_training = False,
                )
                _, log_likelihood_test, _, _ = self.metrics._elbo_fsvi_classification(
                    params,
                    params_feature,
                    state,
                    _prior_mean,
                    _prior_cov,
                    inputs_test,
                    targets_test,
                    inducing_inputs,
                    rng_key,
                    is_training = False,
                )
            if "tdvi" in self.model_type:
                inducing_inputs = inducing_input_fn(inputs_train, rng_key)
                _prior_mean, _prior_cov = prior_fn(
                    inducing_inputs=inducing_inputs,
                    model_params=None,  # TODO: Fix this in case model_params are needed
                    rng_key=rng_key,
                    prior_cov=dtype_default(prior_cov),
                    state=state,
                )

                # _prior_cov = jnp.ones_like(prior_cov) * 1e-4

                _, log_likelihood_train, kl, scale = self.metrics._elbo_tdvi_classification(
                    params,
                    params_feature,
                    state,
                    _prior_mean,
                    _prior_cov,
                    inputs_train,
                    targets_train,
                    inducing_inputs,
                    rng_key,
                    is_training = False,
                )
                _, log_likelihood_test, _, _ = self.metrics._elbo_tdvi_classification(
                    params,
                    params_feature,
                    state,
                    _prior_mean,
                    _prior_cov,
                    inputs_test,
                    targets_test,
                    inducing_inputs,
                    rng_key,
                    is_training = False,
                )
            elif "mfvi" in self.model_type:
                prior_mean, prior_cov = dtype_default(prior_mean), dtype_default(prior_cov)
                _, log_likelihood_train, kl, scale = self.metrics._elbo_mfvi_classification(
                    params,
                    params_feature,
                    state,
                    jnp.ones([1,1], dtype=dtype_default) * prior_mean,
                    jnp.ones([1,1], dtype=dtype_default) * prior_cov,
                    inputs_train,
                    targets_train,
                    None,
                    rng_key,
                    is_training = False,
                )
                _, log_likelihood_test, _, _ = self.metrics._elbo_mfvi_classification(
                    params,
                    params_feature,
                    state,
                    jnp.ones([1,1], dtype=dtype_default) * prior_mean,
                    jnp.ones([1,1], dtype=dtype_default) * prior_cov,
                    inputs_test,
                    targets_test,
                    None,
                    rng_key,
                    is_training = False,
                )
            elif "map" in self.model_type:
                prior_mean, prior_cov = dtype_default(prior_mean), dtype_default(prior_cov)
                loss_train = self.metrics._map_loss_classification(
                    params,
                    params_feature,
                    state,
                    jnp.ones([1,1], dtype=dtype_default) * prior_mean,
                    jnp.ones([1,1], dtype=dtype_default) * prior_cov,
                    inputs_train,
                    targets_train,
                    None,
                    rng_key,
                    is_training = False,
                )
                loss_test = self.metrics._map_loss_classification(
                    params,
                    params_feature,
                    state,
                    jnp.ones([1,1], dtype=dtype_default) * prior_mean,
                    jnp.ones([1,1], dtype=dtype_default) * prior_cov,
                    inputs_test,
                    targets_test,
                    None,
                    rng_key,
                    is_training = False,
                )
                log_likelihood_train, kl, scale = loss_train, 0, 0
                log_likelihood_test = loss_test, 0
            else:
                log_likelihood_train, kl, scale = 0, 0, 0
                log_likelihood_test = 0, 0

            log_likelihoods_train.append(log_likelihood_train)
            log_likelihoods_test.append(log_likelihood_test)
            kls.append(kl)

        loss['elbo_train'] = np.sum(log_likelihoods_train) - kl
        loss['elbo_test'] = np.sum(log_likelihoods_test) - kl
        loss['log_likelihood_train'] = np.sum(log_likelihoods_train)
        loss['log_likelihood_test'] = np.sum(log_likelihoods_test)
        loss['kl'] = np.mean(kls)
        loss['scale'] = np.mean(scale)

        return loss

    def save_results_large(
        self,
        epoch,
        log_likelihood_train,
        accuracy_train,
        log_likelihood_test,
        accuracy_test,
        elbo,
        fKL,
        entropy_test,
        entropy_ood,
        entropy_noise,
        auroc_entropy,
        auroc_expected_entropy,
        auroc_predictive_variance,
        grads_log_var_log_likelihood_mean,
        grads_mean_log_likelihood_mean,
        grads_log_var_function_kl_mean,
        grads_mean_function_kl_mean,
        time_ep,
        time_eval,
    ):
        file_name = f"{self.save_path}/results.csv"
        with open(file_name, "a") as metrics_file:
            metrics_header = [
                "Epoch",
                "Train Negative Log-Likelihood",
                "Test Negative Log-Likelihood",
                "Train Accuracy",
                "Test Accuracy",
                "ELBO",
                "Function KL",
                "Entropy Test",
                "Entropy OOD",
                "Entropy Noise",
                "OOD AUROC (Entropy)",
                "OOD AUROC (Expected Entropy)",
                "OOD AUROC (Predictive Variance)",
                "Grad-Log-Like-Log-Var",
                "Grad-Log-Like-Mean",
                "Grad-Function-KL-Log-Var",
                "Grad-Function-KL-Mean",
                "Time (Epoch)",
                "Time (Evaluation)",
            ]
            writer = csv.DictWriter(metrics_file, fieldnames=metrics_header)
            if os.stat(file_name).st_size == 0:
                writer.writeheader()
            writer.writerow(
                {
                    "Epoch": epoch + 1,
                    "Train Negative Log-Likelihood": -log_likelihood_train,
                    "Test Negative Log-Likelihood": -log_likelihood_test,
                    "Train Accuracy": accuracy_train,
                    "Test Accuracy": accuracy_test,
                    "ELBO": elbo,
                    "Function KL": fKL,
                    "Entropy Test": entropy_test,
                    "Entropy OOD": entropy_ood,
                    "Entropy Noise": entropy_noise,
                    "OOD AUROC (Entropy)": auroc_entropy,
                    "OOD AUROC (Expected Entropy)": auroc_expected_entropy,
                    "OOD AUROC (Predictive Variance)": auroc_predictive_variance,
                    "Grad-Log-Like-Log-Var": grads_log_var_log_likelihood_mean,
                    "Grad-Log-Like-Mean": grads_mean_log_likelihood_mean,
                    "Grad-Function-KL-Log-Var": grads_log_var_function_kl_mean,
                    "Grad-Function-KL-Mean": grads_mean_function_kl_mean,
                    "Time (Epoch)": time_ep,
                    "Time (Evaluation)": time_eval,
                }
            )
            metrics_file.close()

    def save_results(
        self,
        epoch,
        log_likelihood_train,
        log_likelihood_test,
        accuracy_train,
        accuracy_test,
        entropy_test,
        entropy_ood,
        auroc_entropy,
        auroc_expected_entropy,
        auroc_predictive_variance,
        ece_test,
        time_ep,
        time_eval,
    ):
        file_name = f"{self.save_path}/results.csv"
        with open(file_name, "a") as metrics_file:
            metrics_header = [
                "Epoch",
                "Train Log-Likelihood",
                "Test Log-Likelihood",
                "Train Negative Log-Likelihood",
                "Test Negative Log-Likelihood",
                "Train Accuracy",
                "Test Accuracy",
                "Test Entropy",
                "OOD Entropy",
                "OOD AUROC (Entropy)",
                "OOD AUROC (Expected Entropy)",
                "OOD AUROC (Predictive Variance)",
                "Test ECE",
                "Time (Epoch)",
                "Time (Evaluation)",
            ]
            writer = csv.DictWriter(metrics_file, fieldnames=metrics_header)
            if os.stat(file_name).st_size == 0:
                writer.writeheader()
            writer.writerow(
                {
                    "Epoch": epoch + 1,
                    "Train Log-Likelihood": log_likelihood_train,
                    "Test Log-Likelihood": log_likelihood_test,
                    "Train Negative Log-Likelihood": -log_likelihood_train,
                    "Test Negative Log-Likelihood": -log_likelihood_test,
                    "Train Accuracy": accuracy_train,
                    "Test Accuracy": accuracy_test,
                    "Test Entropy": entropy_test,
                    "OOD Entropy": entropy_ood,
                    "OOD AUROC (Entropy)": auroc_entropy,
                    "OOD AUROC (Expected Entropy)": auroc_expected_entropy,
                    "OOD AUROC (Predictive Variance)": auroc_predictive_variance,
                    "Test ECE": ece_test,
                    "Time (Epoch)": time_ep,
                    "Time (Evaluation)": time_eval,
                }
            )
            metrics_file.close()

    def save_results_to_wandb(
        self,
        epoch,
        loss,
        log_likelihood_train,
        log_likelihood_test,
        accuracy_train,
        accuracy_test,
        entropy_test,
        ood_entropy_list,
        auroc_entropy_list,
        auroc_expected_entropy_list,
        auroc_predictive_variance_list,
        ece_test,
        time_ep,
        time_eval,
    ):
        results = {
            "Epoch": epoch + 1,
            "ELBO": loss["elbo_train"],
            "Expected Log-Likelihood": loss["log_likelihood_train"],
            "ELBO (Test)": loss["elbo_test"],
            "Expected Log-Likelihood (Test)": loss["log_likelihood_test"],
            "KL": loss["kl"],
            "KL (Scaled)": loss["scale"] * loss["kl"],
            "Train Log-Likelihood": log_likelihood_train,
            "Test Log-Likelihood": log_likelihood_test,
            "Train Negative Log-Likelihood": -log_likelihood_train,
            "Test Negative Log-Likelihood": -log_likelihood_test,
            "Train Accuracy": accuracy_train,
            "Test Accuracy": accuracy_test,
            "Test Entropy": entropy_test,
            "Test ECE": ece_test,
            "Time (Epoch)": time_ep,
            "Time (Evaluation)": time_eval,
        }

        datasets = ast.literal_eval(self.task.split("[", 1)[1].split("]", 1)[0])
        datasets = [datasets] if type(datasets) is not tuple else datasets

        for i, dataset in enumerate(datasets):
            results[f"OOD Entropy {dataset}"] = ood_entropy_list[i]
            results[f"OOD AUROC {dataset} (Entropy)"] = auroc_entropy_list[i]
            results[f"OOD AUROC {dataset} (Expected Entropy)"] = auroc_expected_entropy_list[i]
            results[f"OOD AUROC {dataset} (Predictive Variance)"] = auroc_predictive_variance_list[i]

        wandb.log(results)

    def save_parameters(self, epoch, params, state):
        path = f"{self.save_path}/chkpts"
        Path(path).mkdir(exist_ok=True)
        if (epoch + 1) % 10 == 0:
            with open(f"{path}/params_{epoch+1}", "wb") as file:
                pickle.dump(params, file)
            with open(f"{path}/state_{epoch+1}", "wb") as file:
                pickle.dump(state, file)
