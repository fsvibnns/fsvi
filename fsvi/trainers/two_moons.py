import getpass
import os
import pickle
import random as random_py

import jax
import jax.numpy as jnp
import numpy as np
import optax
import sklearn
import sklearn.datasets
from jax import jit, random
from tensorflow_probability.substrates import jax as tfp
from tqdm import tqdm

tfd = tfp.distributions

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
path = dname + "/../.."
# print(f'Setting working directory to {path}\n')
os.chdir(path)

from fsvi.utils import utils
from fsvi.utils import utils_logging
from fsvi.utils import utils_training

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

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
    plt.rcParams["font.serif"] = "cm"
    plt.rcParams[
        "text.latex.preamble"
    ] = "\\usepackage{subdepth} \\usepackage{amsfonts} \\usepackage{type1cm}"

dtype_default = jnp.float32
# np.set_printoptions(suppress=True)
eps = 1e-6


def classification_two_moons(
    prior_mean,
    prior_cov,
    epochs,
    logging_frequency,
    seed,
    save_path,
    save,
    **kwargs,
):
    rng_key = random.PRNGKey(seed)
    rng_key, _ = random.split(rng_key)
    random_py.seed(seed)
    np.random.seed(seed)

    # LOAD DATA
    # TODO: move to datasets
    x_train, y_train = sklearn.datasets.make_moons(
        n_samples=200, shuffle=True, noise=0.2, random_state=seed
    )
    y_train = utils._one_hot(y_train, 2)

    h = 0.25
    test_lim = 3
    # x_min, x_max = x_train[:, 0].min() - test_lim, x_train[:, 0].max() + test_lim
    # y_min, y_max = x_train[:, 1].min() - test_lim, x_train[:, 1].max() + test_lim
    x_min, x_max = x_train[:, 0].min() - test_lim, x_train[:, 0].max() + test_lim
    y_min, y_max = x_train[:, 1].min() - test_lim, x_train[:, 1].max() + test_lim
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    x_test = np.vstack((xx.reshape(-1), yy.reshape(-1))).T

    h = 0.25
    test_lim = 7
    x_wide_min, x_wide_max = (
        x_train[:, 0].min() - test_lim,
        x_train[:, 0].max() + test_lim,
    )
    y_wide_min, y_wide_max = (
        x_train[:, 1].min() - test_lim,
        x_train[:, 1].max() + test_lim,
    )
    xx_wide, yy_wide = np.meshgrid(
        np.arange(x_wide_min, x_wide_max, h), np.arange(y_wide_min, y_wide_max, h)
    )
    x_test_wide = np.vstack((xx_wide.reshape(-1), yy_wide.reshape(-1))).T

    input_dim = x_train.shape[-1]
    output_dim = y_train.shape[-1]
    n_train = x_train.shape[0]

    trainloader = None  # TODO: set
    input_shape = None  # TODO: set
    n_batches = None  # TODO: set
    x_train_permuted = None  # TODO: set
    val_frac = None  # TODO: set

    # INITIALIZE TRAINING CLASS
    training = utils_training.Training(
        input_shape=input_shape,
        output_dim=output_dim,
        n_train=n_train,
        n_batches=n_batches,
        **kwargs,
    )

    # INITIALIZE MODEL
    model, init_fn, apply_fn, state, params = training.initialize_model(
        rng_key=rng_key,
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
        x_ood=None,
    )

    # INITIALIZE LOGGING CLASS
    epoch_start = 0
    logging = utils_logging.Logging(
        model=model,
        metrics=metrics,
        loss=loss,
        kl_evaluation=kl_evaluation,
        log_likelihood_evaluation=log_likelihood_evaluation,
        task_evaluation=task_evaluation,
        epoch_start=epoch_start,
        val_frac=val_frac,
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
        rng,
    ):
        rng_key, _ = random.split(rng)
        grads = jax.grad(loss, argnums=0)(
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
        return new_params, opt_state

    print(f"\n--- Training for {epochs} epochs ---\n")
    for epoch in tqdm(range(epochs)):
        for i, data in enumerate(trainloader, 0):
            rng_key, subkey = random.split(rng_key)

            x_batch, y_batch = utils.get_minibatch(
                data, output_dim, input_shape, prediction_type
            )
            inducing_inputs = inducing_input_fn(x_batch, rng_key)
            prior_mean, prior_cov = prior_fn(inducing_inputs)

            params, opt_state = update(
                params,
                state,
                opt_state,
                prior_mean,
                prior_cov,
                x_batch,
                y_batch,
                inducing_inputs,
                rng_key,
            )

        # TODO: update and move to separate logging method
        if epoch % logging_frequency == 0 and epoch > 0:
            rng_key, subkey = random.split(rng_key)
            _, preds_y_mean, preds_y_var = model.predict_y_multisample(
                params, state, x_train, rng_key, 50
            )
            loss = (y_train * jnp.log(preds_y_mean + eps)).sum()
            print(f"Epoch {epoch} : {loss}")
            losses.append(loss)

            plt.figure(figsize=(10, 7))
            plt.plot(range(len(losses)), losses, label="FSVI ELBO")
            plt.legend()
            plt.show()

            _, preds_y_mean, preds_y_var = model.predict_y_multisample(
                params, state, x_test, rng_key, 50
            )
            prediction_mean = preds_y_mean[:, 0].reshape(xx.shape)
            plt.figure(figsize=(10, 7))
            cbar = plt.contourf(xx, yy, prediction_mean, levels=20, cmap=cm.coolwarm)
            cb = plt.colorbar(cbar,)
            cb.ax.set_ylabel(
                "$\mathbb{E}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$",
                rotation=270,
                labelpad=40,
                size=30,
            )
            # cb.ax.set_ylabel('$\mathbb{E}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$', labelpad=-90)
            cb.ax.tick_params(labelsize=30)
            plt.scatter(
                x_train[y_train[:, 0] == 0, 0],
                x_train[y_train[:, 0] == 0, 1],
                color="cornflowerblue",
                edgecolors="black",
            )
            plt.scatter(
                x_train[y_train[:, 0] == 1, 0],
                x_train[y_train[:, 0] == 1, 1],
                color="tomato",
                edgecolors="black",
            )
            plt.tick_params(labelsize=30)
            plt.savefig(
                f"figures/two_moons/two_moons_fsvi_predictive_mean.pdf",
                bbox_inches="tight",
            )
            plt.show()

            prediction_var = preds_y_var[:, 0].reshape(xx.shape)
            plt.figure(figsize=(10, 7))
            cbar = plt.contourf(xx, yy, prediction_var, levels=16, cmap=cm.Greys)
            cb = plt.colorbar(cbar,)
            cb.ax.set_ylabel(
                "$\mathbb{V}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$",
                rotation=270,
                labelpad=40,
                size=30,
            )
            # cb.ax.set_ylabel('$V[\mathbf{y} | \mathcal{D}; \mathbf{x}]$', labelpad=-90)
            cb.ax.tick_params(labelsize=30)
            plt.scatter(
                x_train[y_train[:, 0] == 0, 0],
                x_train[y_train[:, 0] == 0, 1],
                color="cornflowerblue",
                edgecolors="black",
            )
            plt.scatter(
                x_train[y_train[:, 0] == 1, 0],
                x_train[y_train[:, 0] == 1, 1],
                color="tomato",
                edgecolors="black",
            )
            plt.tick_params(labelsize=30)
            plt.savefig(
                f"figures/two_moons/two_moons_fsvi_predictive_variance.pdf",
                bbox_inches="tight",
            )
            plt.show()

    #%%

    _, preds_y_mean, preds_y_var = model.predict_y_multisample(
        params, state, x_test, rng_key, 50
    )

    prediction_mean = preds_y_mean[:, 0].reshape(xx.shape)
    plt.figure(figsize=(10, 7))
    cbar = plt.contourf(xx, yy, prediction_mean, levels=20, cmap=cm.coolwarm)
    cb = plt.colorbar(cbar,)
    cb.ax.set_ylabel(
        "$\mathbb{E}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$",
        rotation=270,
        labelpad=40,
        size=30,
    )
    # cb.ax.set_ylabel('$E[y | \mathcal{D}; x]$', labelpad=-90)
    cb.ax.tick_params(labelsize=30)
    plt.scatter(
        x_train[y_train[:, 0] == 0, 0],
        x_train[y_train[:, 0] == 0, 1],
        color="cornflowerblue",
        edgecolors="black",
    )
    plt.scatter(
        x_train[y_train[:, 0] == 1, 0],
        x_train[y_train[:, 0] == 1, 1],
        color="tomato",
        edgecolors="black",
    )
    plt.tick_params(labelsize=30)
    plt.savefig(
        f"figures/two_moons/two_moons_fsvi_predictive_mean.pdf", bbox_inches="tight"
    )
    plt.show()

    prediction_var = preds_y_var[:, 0].reshape(xx.shape)
    plt.figure(figsize=(10, 7))
    cbar = plt.contourf(xx, yy, prediction_var, levels=16, cmap=cm.Greys)
    cb = plt.colorbar(cbar,)
    cb.ax.set_ylabel(
        "$\mathbb{V}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$",
        rotation=270,
        labelpad=40,
        size=30,
    )
    # cb.ax.set_ylabel('$V[\mathbf{y} | \mathcal{D}; \mathbf{x}]$', labelpad=-90)
    cb.ax.tick_params(labelsize=30)
    plt.scatter(
        x_train[y_train[:, 0] == 0, 0],
        x_train[y_train[:, 0] == 0, 1],
        color="cornflowerblue",
        edgecolors="black",
    )
    plt.scatter(
        x_train[y_train[:, 0] == 1, 0],
        x_train[y_train[:, 0] == 1, 1],
        color="tomato",
        edgecolors="black",
    )
    plt.tick_params(labelsize=30)
    plt.savefig(
        f"figures/two_moons/two_moons_fsvi_predictive_variance.pdf", bbox_inches="tight"
    )
    plt.show()

    _, preds_y_mean, preds_y_var = model.predict_y_multisample(
        params, state, x_test_wide, rng_key, 50
    )

    prediction_mean = preds_y_mean[:, 0].reshape(xx_wide.shape)
    plt.figure(figsize=(10, 7))
    cbar = plt.contourf(xx_wide, yy_wide, prediction_mean, levels=20, cmap=cm.coolwarm)
    cb = plt.colorbar(cbar,)
    cb.ax.set_ylabel(
        "$\mathbb{E}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$",
        rotation=270,
        labelpad=40,
        size=30,
    )
    # cb.ax.set_ylabel('$E[y | \mathcal{D}; x]$', labelpad=-90)
    cb.ax.tick_params(labelsize=30)
    plt.scatter(
        x_train[y_train[:, 0] == 0, 0],
        x_train[y_train[:, 0] == 0, 1],
        color="cornflowerblue",
        edgecolors="black",
    )
    plt.scatter(
        x_train[y_train[:, 0] == 1, 0],
        x_train[y_train[:, 0] == 1, 1],
        color="tomato",
        edgecolors="black",
    )
    plt.tick_params(labelsize=30)
    plt.savefig(
        f"figures/two_moons/two_moons_fsvi_predictive_mean_wide.pdf",
        bbox_inches="tight",
    )
    plt.show()

    prediction_var = preds_y_var[:, 0].reshape(xx_wide.shape)
    plt.figure(figsize=(10, 7))
    cbar = plt.contourf(xx_wide, yy_wide, prediction_var, levels=16, cmap=cm.Greys)
    cb = plt.colorbar(cbar,)
    cb.ax.set_ylabel(
        "$\mathbb{V}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$",
        rotation=270,
        labelpad=40,
        size=30,
    )
    # cb.ax.set_ylabel('$V[y | \mathcal{D}; x]$', labelpad=-90)
    cb.ax.tick_params(labelsize=30)
    plt.scatter(
        x_train[y_train[:, 0] == 0, 0],
        x_train[y_train[:, 0] == 0, 1],
        color="cornflowerblue",
        edgecolors="black",
    )
    plt.scatter(
        x_train[y_train[:, 0] == 1, 0],
        x_train[y_train[:, 0] == 1, 1],
        color="tomato",
        edgecolors="black",
    )
    plt.tick_params(labelsize=30)
    plt.savefig(
        f"figures/two_moons/two_moons_fsvi_predictive_variance_wide.pdf",
        bbox_inches="tight",
    )
    plt.show()

    #%%

    ####### Saving model parameters #######
    if save:
        with open(f"{save_path}/params_pickle", "wb") as file:
            pickle.dump(params, file)

        plt.savefig(
            f"{save_path}/two_moons_fsvi_predictive_mean.pdf", bbox_inches="tight"
        )
        plt.savefig(
            f"{save_path}/two_moons_fsvi_predictive_variance.pdf", bbox_inches="tight"
        )
        plt.savefig(
            f"{save_path}/two_moons_fsvi_predictive_mean_wide.pdf", bbox_inches="tight"
        )
        plt.savefig(
            f"{save_path}/two_moons_fsvi_predictive_variance_wide.pdf",
            bbox_inches="tight",
        )


# TODO: update
def two_moons_ensemble_mlp(
    task,
    data_training,
    data_ood,
    model_type,
    architecture,
    activation,
    prior_mean,
    prior_cov,
    prior_type,
    batch_size,
    epochs,
    learning_rate,
    regularization,
    inducing_points,
    inducing_type,
    kl_scale,
    full_cov,
    n_samples,
    tau,
    ind_lim,
    logging_frequency,
    figsize,
    seed,
    save_path,
    save,
    resume,
    debug,
):
    rng_key = random.PRNGKey(seed)
    rng_key, _ = random.split(rng_key)

    # >>>>>>>>>>>>>>>>>>> Model-specific setup begins below

    x_train, y_train = sklearn.datasets.make_moons(
        n_samples=200, shuffle=True, noise=0.2, random_state=seed
    )
    y_train = utils._one_hot(y_train, 2)

    h = 0.25
    test_lim = 3
    # x_min, x_max = x_train[:, 0].min() - test_lim, x_train[:, 0].max() + test_lim
    # y_min, y_max = x_train[:, 1].min() - test_lim, x_train[:, 1].max() + test_lim
    x_min, x_max = x_train[:, 0].min() - test_lim, x_train[:, 0].max() + test_lim
    y_min, y_max = x_train[:, 1].min() - test_lim, x_train[:, 1].max() + test_lim
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    x_test = np.vstack((xx.reshape(-1), yy.reshape(-1))).T

    h = 0.25
    test_lim = 7
    x_wide_min, x_wide_max = (
        x_train[:, 0].min() - test_lim,
        x_train[:, 0].max() + test_lim,
    )
    y_wide_min, y_wide_max = (
        x_train[:, 1].min() - test_lim,
        x_train[:, 1].max() + test_lim,
    )
    xx_wide, yy_wide = np.meshgrid(
        np.arange(x_wide_min, x_wide_max, h), np.arange(y_wide_min, y_wide_max, h)
    )
    x_test_wide = np.vstack((xx_wide.reshape(-1), yy_wide.reshape(-1))).T

    if architecture != []:
        for i in range(0, len(architecture)):
            architecture[i] = int(architecture[i])

    # >>>>>>>>>>>>>>>>>>> Model-specific setup begins below

    input_dim = x_train.shape[-1]
    output_dim = y_train.shape[-1]
    n_train = x_train.shape[0]

    stochastic_parameters = False

    model = MLP(
        output_dim=output_dim,
        activation_fn=activation,
        architecture=architecture,
        stochastic_parameters=stochastic_parameters,
    )
    (
        init_fn,
        apply_fn,
    ) = model.forward  # pass apply_fn as first argument to objectives below

    metrics = Objectives(
        model=apply_fn,  # apply_fn from above
        predict_f=model.predict_f,
        predict_y=model.predict_y,
        predict_f_multisample=model.predict_f_multisample,
        predict_f_multisample_jitted=model.predict_f_multisample_jitted,
        predict_y_multisample=model.predict_y_multisample,
        predict_y_multisample_jitted=model.predict_y_multisample_jitted,
        output_dim=output_dim,
        kl_scale=kl_scale,
        full_cov=full_cov,
    )

    loss = metrics.map_loss_classification
    accuracy = metrics.accuracy

    inducing_inputs = jax.random.uniform(
        rng_key, [inducing_points, input_dim], dtype_default, -ind_lim, ind_lim
    )
    if prior_type == "bnn_induced":
        rng_key0, _ = random.split(rng_key)
        prior_mean_params, prior_cov_params = prior_mean, prior_cov
        induced_prior_fn = lambda params, inducing_inputs: utils_linearization.induced_prior_fn(
            model,
            params,
            state,
            prior_mean_params,
            prior_cov_params,
            output_dim,
            inducing_inputs,
            rng_key0,
        )
    else:
        prior_mean = jnp.zeros(inducing_points) * dtype_default(prior_mean)
        prior_cov = jnp.ones(inducing_points) * dtype_default(prior_cov)

    for ensemble_num in range(5):
        # Initialize NN
        rng_key, _ = random.split(rng_key)
        opt = optax.adam(learning_rate)
        x_batch = x_train[0, :]
        params_init, state = model.initialize(rng_key, x_batch)
        opt_state = opt.init(params_init)
        params = params_init

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
            rng,
        ):
            rng_key, _ = random.split(rng)
            params_copy = params
            grads = jax.grad(loss, argnums=0)(
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
            return new_params, opt_state

        #%%

        ind_lim = 10
        print(f"\n--- Training for {epochs} epochs ---\n")
        for epoch in range(epochs):
            rng_key, subkey = random.split(rng_key)
            permutation = np.random.permutation(n_train)
            x_train_permuted = x_train[permutation, :]
            random_samples = jax.random.uniform(
                rng_key, [int(inducing_points / 2), 2], dtype_default, -ind_lim, ind_lim
            )
            training_samples = x_train_permuted[: int(inducing_points / 2), :]
            inducing_inputs = jnp.concatenate([random_samples, training_samples], 0)
            params, opt_state = update(
                params,
                state,
                opt_state,
                prior_mean,
                prior_cov,
                x_train,
                y_train,
                inducing_inputs,
                rng_key,
            )

        # if epoch % logging_frequency == 0:
        #     rng_key, subkey = random.split(rng_key)
        #     _, preds_y_mean, preds_y_var = model.predict_y_multisample(params, state, x_train, rng_key, n_samples=50)
        #     LOSS = (y_train * jnp.log(preds_y_mean + eps)).sum()
        #     print(f"Epoch {epoch} : {LOSS}")

        #     _, preds_y_mean, preds_y_var = model.predict_y_multisample(params, state, x_test, rng_key)

        #     prediction_mean = preds_y_mean[:, 0].reshape(xx.shape)
        #     plt.figure(figsize=(10, 7))
        #     cbar = plt.contourf(xx, yy, prediction_mean, levels=20, cmap=cm.coolwarm)
        #     cb = plt.colorbar(cbar, )
        #     cb.ax.set_ylabel('$\mathbb{E}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$', rotation=270, labelpad=40, size=30)
        #     # cb.ax.set_ylabel('$\mathbb{E}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$', labelpad=-90)
        #     cb.ax.tick_params(labelsize=30)
        #     plt.scatter(x_train[y_train[:, 0] == 0, 0], x_train[y_train[:, 0] == 0, 1],
        #                 color='cornflowerblue',
        #                 edgecolors='black')
        #     plt.scatter(x_train[y_train[:, 0] == 1, 0], x_train[y_train[:, 0] == 1, 1],
        #                 color='tomato',
        #                 edgecolors='black')
        #     plt.tick_params(labelsize=30)
        #     plt.savefig(f'figures/two_moons/two_moons_ensemble_predictive_mean.pdf', bbox_inches='tight')
        #     plt.show()

        #     prediction_var = preds_y_var[:, 0].reshape(xx.shape)
        #     plt.figure(figsize=(10, 7))
        #     cbar = plt.contourf(xx, yy, prediction_var, levels=16, cmap=cm.Greys)
        #     cb = plt.colorbar(cbar, )
        #     cb.ax.set_ylabel('$\mathbb{V}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$', rotation=270, labelpad=40, size=30)
        #     # cb.ax.set_ylabel('$V[\mathbf{y} | \mathcal{D}; \mathbf{x}]$', labelpad=-90)
        #     cb.ax.tick_params(labelsize=30)
        #     plt.scatter(x_train[y_train[:, 0] == 0, 0], x_train[y_train[:, 0] == 0, 1], color='cornflowerblue',
        #                 edgecolors='black')
        #     plt.scatter(x_train[y_train[:, 0] == 1, 0], x_train[y_train[:, 0] == 1, 1], color='tomato',
        #                 edgecolors='black')
        #     plt.tick_params(labelsize=30)
        #     plt.savefig(f'figures/two_moons/two_moons_ensemble_predictive_variance.pdf', bbox_inches='tight')
        #     plt.show()

        #%%

        _, preds_y_mean, preds_y_var = model.predict_y_multisample(
            params, state, x_test, rng_key, 50
        )

        prediction_mean = preds_y_mean[:, 0].reshape(xx.shape)
        plt.figure(figsize=(10, 7))
        cbar = plt.contourf(xx, yy, prediction_mean, levels=20, cmap=cm.coolwarm)
        cb = plt.colorbar(cbar,)
        cb.ax.set_ylabel(
            "$\mathbb{E}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$",
            rotation=270,
            labelpad=40,
            size=30,
        )
        # cb.ax.set_ylabel('$E[y | \mathcal{D}; x]$', labelpad=-90)
        cb.ax.tick_params(labelsize=30)
        plt.scatter(
            x_train[y_train[:, 0] == 0, 0],
            x_train[y_train[:, 0] == 0, 1],
            color="cornflowerblue",
            edgecolors="black",
        )
        plt.scatter(
            x_train[y_train[:, 0] == 1, 0],
            x_train[y_train[:, 0] == 1, 1],
            color="tomato",
            edgecolors="black",
        )
        plt.tick_params(labelsize=30)
        plt.show()

        prediction_var = preds_y_var[:, 0].reshape(xx.shape)
        plt.figure(figsize=(10, 7))
        cbar = plt.contourf(xx, yy, prediction_var, levels=16, cmap=cm.Greys)
        cb = plt.colorbar(cbar,)
        cb.ax.set_ylabel(
            "$\mathbb{V}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$",
            rotation=270,
            labelpad=40,
            size=30,
        )
        # cb.ax.set_ylabel('$V[\mathbf{y} | \mathcal{D}; \mathbf{x}]$', labelpad=-90)
        cb.ax.tick_params(labelsize=30)
        plt.scatter(
            x_train[y_train[:, 0] == 0, 0],
            x_train[y_train[:, 0] == 0, 1],
            color="cornflowerblue",
            edgecolors="black",
        )
        plt.scatter(
            x_train[y_train[:, 0] == 1, 0],
            x_train[y_train[:, 0] == 1, 1],
            color="tomato",
            edgecolors="black",
        )
        plt.tick_params(labelsize=30)
        plt.show()

        _, preds_y_mean, preds_y_var = model.predict_y_multisample(
            params, state, x_test_wide, rng_key, 50
        )

        prediction_mean = preds_y_mean[:, 0].reshape(xx_wide.shape)
        plt.figure(figsize=(10, 7))
        cbar = plt.contourf(
            xx_wide, yy_wide, prediction_mean, levels=20, cmap=cm.coolwarm
        )
        cb = plt.colorbar(cbar,)
        cb.ax.set_ylabel(
            "$\mathbb{E}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$",
            rotation=270,
            labelpad=40,
            size=30,
        )
        # cb.ax.set_ylabel('$E[y | \mathcal{D}; x]$', labelpad=-90)
        cb.ax.tick_params(labelsize=30)
        plt.scatter(
            x_train[y_train[:, 0] == 0, 0],
            x_train[y_train[:, 0] == 0, 1],
            color="cornflowerblue",
            edgecolors="black",
        )
        plt.scatter(
            x_train[y_train[:, 0] == 1, 0],
            x_train[y_train[:, 0] == 1, 1],
            color="tomato",
            edgecolors="black",
        )
        plt.tick_params(labelsize=30)
        plt.show()

        prediction_var = preds_y_var[:, 0].reshape(xx_wide.shape)
        plt.figure(figsize=(10, 7))
        cbar = plt.contourf(xx_wide, yy_wide, prediction_var, levels=16, cmap=cm.Greys)
        cb = plt.colorbar(cbar,)
        cb.ax.set_ylabel(
            "$\mathbb{V}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$",
            rotation=270,
            labelpad=40,
            size=30,
        )
        # cb.ax.set_ylabel('$V[y | \mathcal{D}; x]$', labelpad=-90)
        cb.ax.tick_params(labelsize=30)
        plt.scatter(
            x_train[y_train[:, 0] == 0, 0],
            x_train[y_train[:, 0] == 0, 1],
            color="cornflowerblue",
            edgecolors="black",
        )
        plt.scatter(
            x_train[y_train[:, 0] == 1, 0],
            x_train[y_train[:, 0] == 1, 1],
            color="tomato",
            edgecolors="black",
        )
        plt.tick_params(labelsize=30)
        plt.show()

        with open(f"params_pickle_{ensemble_num}", "wb") as file:
            pickle.dump(params, file)

        print(f"Ensemble Tasks : {ensemble_num}/5 finished!")

    prediction_mean_list = []
    prediction_mean_wide_list = []
    for ensemble_num in range(5):
        with open(f"params_pickle_{ensemble_num}", "rb") as file:
            params_tp = pickle.load(file)
        _, preds_y_mean, _ = model.predict_y_multisample(
            params_tp, state, x_test, rng_key, 50
        )
        _, preds_y_mean_wide, _ = model.predict_y_multisample(
            params_tp, state, x_test_wide, rng_key, 50
        )
        prediction_mean_list.append(preds_y_mean)
        prediction_mean_wide_list.append(preds_y_mean_wide)

    prediction_mean_ensemble = np.stack(prediction_mean_list, axis=0)
    prediction_mean_wide_ensemble = np.stack(prediction_mean_wide_list, axis=0)

    prediction_mean_ensemble = prediction_mean_ensemble.mean(0)
    prediction_mean_wide_ensemble = prediction_mean_wide_ensemble.mean(0)

    prediction_mean = prediction_mean_ensemble[:, 0].reshape(xx.shape)
    plt.figure(figsize=(10, 7))
    cbar = plt.contourf(xx, yy, prediction_mean, levels=20, cmap=cm.coolwarm)
    cb = plt.colorbar(cbar,)
    cb.ax.set_ylabel(
        "$\mathbb{E}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$",
        rotation=270,
        labelpad=40,
        size=30,
    )
    # cb.ax.set_ylabel('$E[y | \mathcal{D}; x]$', labelpad=-90)
    cb.ax.tick_params(labelsize=30)
    plt.scatter(
        x_train[y_train[:, 0] == 0, 0],
        x_train[y_train[:, 0] == 0, 1],
        color="cornflowerblue",
        edgecolors="black",
    )
    plt.scatter(
        x_train[y_train[:, 0] == 1, 0],
        x_train[y_train[:, 0] == 1, 1],
        color="tomato",
        edgecolors="black",
    )
    plt.tick_params(labelsize=30)
    plt.show()

    prediction_mean_wide = prediction_mean_wide_ensemble[:, 0].reshape(xx_wide.shape)
    plt.figure(figsize=(10, 7))
    cbar = plt.contourf(
        xx_wide, yy_wide, prediction_mean_wide, levels=20, cmap=cm.coolwarm
    )
    cb = plt.colorbar(cbar,)
    cb.ax.set_ylabel(
        "$\mathbb{E}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$",
        rotation=270,
        labelpad=40,
        size=30,
    )
    # cb.ax.set_ylabel('$E[y | \mathcal{D}; x]$', labelpad=-90)
    cb.ax.tick_params(labelsize=30)
    plt.scatter(
        x_train[y_train[:, 0] == 0, 0],
        x_train[y_train[:, 0] == 0, 1],
        color="cornflowerblue",
        edgecolors="black",
    )
    plt.scatter(
        x_train[y_train[:, 0] == 1, 0],
        x_train[y_train[:, 0] == 1, 1],
        color="tomato",
        edgecolors="black",
    )
    plt.tick_params(labelsize=30)
    plt.show()

    # plt.savefig(f'two_moons_ensemble_predictive_mean.pdf', bbox_inches='tight')
    # plt.savefig(f'{save_path}/two_moons_ensemble_predictive_variance.pdf', bbox_inches='tight')
    # plt.savefig(f'{save_path}/two_moons_ensemble_predictive_mean_wide.pdf', bbox_inches='tight')
    # plt.savefig(f'{save_path}/two_moons_ensemble_predictive_variance_wide.pdf', bbox_inches='tight')
