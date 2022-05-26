import os
from tqdm import tqdm
import time
import csv
import pickle
import getpass
from scipy.stats import norm
from jax.scipy.special import logsumexp
import math
import seqtools

import jax
import jax.numpy as jnp
from jax import jit, grad, random
from jax.tree_util import tree_flatten
import haiku as hk

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import optax
import numpy as np

import pandas as pd
import joypy

try:
    from bayesian_benchmarks.data import get_regression_data
    from bayesian_benchmarks.database_utils import Database
except:
    print('WARNING: bayesian_benchmarks could not be loaded.')

import uncertainty_metrics.numpy as um
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 75
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.figsize'] = 6, 4
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['grid.linestyle'] = "-"
plt.rcParams['grid.linewidth'] = 1.0
plt.rcParams['legend.facecolor'] = 'white'
# plt.rcParams['grid.color'] = "grey"
if getpass.getuser() == 'ANON':
    plt.rcParams['text.usetex'] = True
    # plt.rcParams['font.family'] = "normal"
    # plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.family'] = "serif"
    plt.rcParams['font.serif'] = "cm"
    plt.rcParams['text.latex.preamble'] = "\\usepackage{subdepth} \\usepackage{amsfonts} \\usepackage{type1cm}"
    from matplotlib.backends.backend_pdf import PdfPages

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
path = dname + '/../..'
# print(f'Setting working directory to {path}\n')
os.chdir(path)

from fsvi.utils import utils, utils_linearization
from fsvi.utils import datasets
from fsvi.models.networks import CNN, ACTIVATION_DICT
from fsvi.models.networks import MLP as MLP
from fsvi.utils.objectives import Objectives_hk as Objectives

dtype_default = jnp.float32
# np.set_printoptions(suppress=True)
eps = 1e-6

res = {}

res['epoch'] = []
res['train_loglik'] = []

res['train_rmse'] = []
res['train_rmse_unnormalized'] = []

res['test_loglik'] = []
res['test_loglik_unnormalized'] = []

res['test_mae'] = []
res['test_mae_unnormalized'] = []

res['test_rmse'] = []
res['test_rmse_unnormalized'] = []

res['elbo'] = []
res['log_likelihood'] = []
res['kl'] = []
res['pred_var_train'] = []
res['pred_var_test'] = []
res['var_params'] = []
res['var_params_median'] = []
res['var_params_max'] = []


def regression_synthetic_1d(
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

    dataset = kwargs['task'].split("uci_", 1)[1].split("_", 1)[0]
    kwargs['data_training'] = 'uci'
    data = get_regression_data(dataset, split=seed)
    x_train, y_train, x_test, y_test = data.X_train, data.Y_train, data.X_test, data.Y_test
    x, y = x_train, y_train
    noise_std = 2.
    x_test_features = x_test

    x_mean = data.X_mean
    x_std = data.X_std
    y_mean = data.Y_mean
    y_std = data.Y_std

    if dataset == 'boston':  # MC Dropout | RMSE: 2.80 ± 0.19 | Log-likelihood: -2.39 ± 0.05
        tau = tau  # 0.2  # 0.1, 0.15, 0.2
        ind_lim = ind_lim  # 10
        ''' RESULTS
        ++ CONFIG
            --activation relu
            --learning_rate 5e-4
            --prior_mean 0.
            --prior_cov 1e3
            --inducing_points 2
            --kl_scale none
            full_cov = True
            tau = 0.2
            ind_lim = 10
            n_samples = 10
            seed = 0
            jitter = 1e-3
        ++ RESULT (does NOT beat MC Dropout) (not fully converged, later iterations were a bit worse)
            Epoch: 100000
            Train RMSE: 0.64  |  Train Standardized RMSE: 0.07  |  Train Log-Likelihood: -1.76
            Test RMSE: 2.92  |  Test Standardized RMSE: 0.32  |  Test Log-Likelihood: -2.54
            ELBO: -786  |  Log-Likelihood: -784  |  KL: 2
        ++ OLD but better:
        ++ CONFIG
            --activation relu
            --learning_rate 1e-3
            --prior_cov 1e5
            --inducing_points 10
            full_cov = False
            tau = 0.2
            ind_lim = 1000
            seed = 0
            jitter = 0 or 1e-2 or 1e-3 (not sure)
        ++ RESULT
            Epoch: 50000
            Train RMSE: 3.06  |  Train Standardized RMSE: 0.33  |  Train Log-Likelihood: -2.51
            Test RMSE: 2.83  |  Test Standardized RMSE: 0.31  |  Test Log-Likelihood: -2.47
            ELBO: -1198  |  Log-Likelihood: -790  |  KL: 9
        '''
    elif dataset == 'concrete':  # MC Dropout | RMSE: 4.81 ± 0.14 | Log-likelihood: -2.94 ± 0.02
        tau = tau  # 0.1  # 0.025, 0.05, 0.075
        ind_lim = ind_lim  # 2
        ''' RESULTS
        ++ CONFIG
            --activation relu
            --learning_rate 5e-4
            --prior_mean 0.
            --prior_cov 10
            --inducing_points 2
            --kl_scale none
            --tau 0.1
            --ind_lim 5
            --n_samples 10
            full_cov = False
            seed = 0
            jitter = 1e-3
            initialization = [-10.0, -8.0]
        ++ RESULT (beats MC Dropout in terms of RMSE)
            Epoch: 180000
            Train RMSE: 2.60  |  Train Standardized RMSE: 0.16  |  Train Log-Likelihood: -2.35
            Test RMSE: 4.58  |  Test Standardized RMSE: 0.27  |  Test Log-Likelihood: -3.16
            ELBO: -1818  |  Log-Likelihood: -1817  |  KL: 1
        ++ CONFIG
            --activation relu
            --learning_rate 5e-4
            --prior_mean 0.
            --prior_cov 1e3
            --inducing_points 2
            --kl_scale none
            full_cov = True
            tau = 0.1
            ind_lim = 10
            n_samples = 10
            seed = 0
            jitter = 1e-3
        ++ RESULT (beats MC Dropout)
            Epoch: 100000
            Train RMSE: 2.59  |  Train Standardized RMSE: 0.16  |  Train Log-Likelihood: -2.38
            Test RMSE: 4.28  |  Test Standardized RMSE: 0.26  |  Test Log-Likelihood: -2.89
            ELBO: -1922  |  Log-Likelihood: -1920  |  KL: 2
        '''
    elif dataset == 'energy':  # MC Dropout | RMSE: 1.09 ± 0.05 | Log-likelihood: -1.72 ± 0.02
        tau = tau  # 0.75  # 0.25, 0.5, 0.75
        ind_lim = ind_lim  # 10
        ''' RESULTS
        ++ CONFIG
            --activation relu
            --learning_rate 5e-4
            --prior_mean 0.
            --prior_cov 10
            --inducing_points 2
            --kl_scale none
            --tau 0.75
            --ind_lim 5
            --n_samples 10
            full_cov = False
            seed = 0
            jitter = 1e-3
            initialization = [-10.0, -8.0]
        ++ RESULT (beats MC Dropout)
            Epoch: 165000
            Train RMSE: 0.30  |  Train Standardized RMSE: 0.03  |  Train Log-Likelihood: -1.10
            Test RMSE: 0.36  |  Test Standardized RMSE: 0.04  |  Test Log-Likelihood: -1.11
            ELBO: -739  |  Log-Likelihood: -735  |  KL: 4
        ++ CONFIG
            --activation relu
            --learning_rate 5e-4
            --prior_mean 0.
            --prior_cov 1e4
            --inducing_points 2
            --kl_scale none
            --n_samples 10
            --tau 0.75
            --ind_lim 10
            full_cov = False
            seed = 0
            jitter = 1e-3
            initialization = [-10.0, -8.0]
        ++ RESULT (beats MC Dropout)
            Epoch: 100000
            Train RMSE: 0.24  |  Train Standardized RMSE: 0.02  |  Train Log-Likelihood: -1.08
            Test RMSE: 0.32  |  Test Standardized RMSE: 0.03  |  Test Log-Likelihood: -1.10
            ELBO: -737  |  Log-Likelihood: -735  |  KL: 2
        '''
    elif dataset == 'kin8nm':  # MC Dropout | RMSE: 0.10 ± 0.00 | Log-likelihood: 0.95 ± 0.01
        tau = 200  # 150, 200, 250
        ind_lim = 10
    elif dataset == 'winered':  # MC Dropout | RMSE: 0.61 ± 0.01 | Log-likelihood: -0.92 ± 0.01
        tau = tau  # 2.0  # 2.5, 3.0, 3.5
        ind_lim = ind_lim  # 10000
        ''' RESULTS
        ++ CONFIG
            --activation relu
            --learning_rate 5e-4
            --prior_mean 0.
            --prior_cov 10
            --inducing_points 100
            --kl_scale equal
            --tau 2.0
            --ind_lim 5
            --n_samples 10
            full_cov = False
            seed = 0
            jitter = 1e-3
            initialization = [-10.0, -8.0]
        ++ RESULT (beats MC Dropout)
            Epoch: 100000
            Train RMSE: 0.52  |  Train Standardized RMSE: 0.64  |  Train Log-Likelihood: -0.83
            Test RMSE: 0.59  |  Test Standardized RMSE: 0.73  |  Test Log-Likelihood: -0.90
            ELBO: -1675  |  Log-Likelihood: -1416  |  KL: 18
        ++ CONFIG
            --activation relu
            --learning_rate 5e-4
            --prior_mean 0.
            --prior_cov 1e4
            --inducing_points 2
            --kl_scale none
            full_cov = True
            tau = 2.0
            ind_lim = 10000
            n_samples = 10
            seed = 0
            jitter = 1e-3
        ++ RESULT (beats MC Dropout)
            Epoch: 50000
            Train RMSE: 0.54  |  Train Standardized RMSE: 0.66  |  Train Log-Likelihood: -0.86
            Test RMSE: 0.56  |  Test Standardized RMSE: 0.70  |  Test Log-Likelihood: -0.89
            ELBO: -1467  |  Log-Likelihood: -1456  |  KL: 11
        ++ CONFIG
            --activation relu
            --learning_rate 5e-4
            --prior_mean 0.
            --prior_cov 1e3
            --inducing_points 2
            --kl_scale none
            full_cov = True
            tau = 2.0
            ind_lim = 100000
            n_samples = 10
            seed = 0
            jitter = 1e-3
        ++ RESULT (beats MC Dropout)
            Epoch: 300000
            Train RMSE: 0.64  |  Train Standardized RMSE: 0.80  |  Train Log-Likelihood: -0.99
            Test RMSE: 0.59  |  Test Standardized RMSE: 0.73  |  Test Log-Likelihood: -0.92
            ELBO: -1737  |  Log-Likelihood: -1735  |  KL: 2
        ++ OLD:
        ++ CONFIG
            --activation relu
            --learning_rate 5e-4
            --prior_cov 1e5
            --inducing_points 10
            --kl_scale equal
            full_cov = False
            tau = 2.0
            ind_lim = 1000
            seed = 0
            jitter = 1e-2
        ++ RESULT
            Epoch: 140000
            Train RMSE: 0.48  |  Train Standardized RMSE: 0.60  |  Train Log-Likelihood: -0.82
            Test RMSE: 0.57  |  Test Standardized RMSE: 0.70  |  Test Log-Likelihood: -0.90
            ELBO: -2627  |  Log-Likelihood: -1395  |  KL: 9
        '''
    else:
        tau = 1.0
        ind_lim = 1000



    # TODO: remove hardcoding
    noise_var = noise_std ** 2

    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    input_shape = [1, input_dim]

    # DEFINE NUMPY TRAINLOADER  # TODO: test implementation
    n_train = x_train.shape[0]
    train_dataset = seqtools.collate([x_train, y_train])
    if batch_size == 0:
        _trainloader = seqtools.batch(train_dataset, n_train, collate_fn=datasets.collate_fn)
    else:
        _trainloader = seqtools.batch(train_dataset, batch_size, collate_fn=datasets.collate_fn)

    trainloader = []
    for i, data in enumerate(_trainloader, 0):
        x_batch = np.array(data[0], dtype=dtype_default)
        y_batch = np.array(data[1], dtype=dtype_default)
        trainloader.append([x_batch, y_batch])

    permutation = np.random.permutation(x_train.shape[0])
    x_train_permuted = x_train[permutation,:]

    n_train = x_train.shape[0]
    batch_size = n_train
    n_batches = np.int(n_train / batch_size)

    val_frac = 0.

    # INITIALIZE TRAINING CLASS
    training = Training(
        input_shape=input_shape,
        output_dim=output_dim,
        n_train=n_train,
        batch_size=batch_size,
        n_batches=n_train // batch_size,
        stochastic_linearization=True,
        full_ntk=False,
        model_type=model_type,
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

    inducing_input_fn, prior_fn = training.kl_input_functions(
        apply_fn= apply_fn,
        predict_f_deterministic=model.predict_f_deterministic,
        state = state,
        params = params,
        prior_mean = prior_mean,
        prior_cov = prior_cov,
        rng_key = rng_key,
        x_ood = [None],
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
        y_train_permuted=None,
        x_test=x_test,
        y_test=None,
        x_ood=[None],
        n_train=n_train,
        val_frac=val_frac,
        epochs=epochs,
        save=save,
        save_path=save_path,
        model_type=model_type,
        **kwargs,
    )

    @jit
    def update(params, state, opt_state, prior_mean, prior_cov, x_batch, y_batch, inducing_inputs, rng):
        rng_key, _ = random.split(rng)
        grads = jax.grad(loss, argnums = 0)(params, state, prior_mean, prior_cov, x_batch, y_batch, inducing_inputs, rng_key)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    print(f'\n--- Training for {epochs} epochs ---\n')
    for epoch in range(epochs):
        logging.t0 = time.time()

        for i, data in enumerate(trainloader, 0):
            rng_key, subkey = random.split(rng_key)

            x_batch, y_batch = utils.get_minibatch(data, output_dim, input_shape, prediction_type)
            inducing_inputs = inducing_input_fn(x_batch, rng_key)
            prior_mean, prior_cov = prior_fn(inducing_inputs)

            params, opt_state = update(params, state, opt_state, prior_mean, prior_cov, x_batch, y_batch, inducing_inputs, rng_key)

        logging.T = time.time()

#%%
        if False:
        # if (epoch+1) % int(logging_frequency) == 0:
            if (epoch+1) % int(logging_frequency * 10) == 0:
                n_samples = 50
            else:
                n_samples = 50
            preds_f_samples_train, preds_f_mean_train, preds_f_var_train = model.predict_f_multisample(params, state, x_train, rng_key, n_samples)
            preds_f_samples_test, preds_f_mean_test, preds_f_var_test = model.predict_f_multisample(params, state, x_test, rng_key, n_samples)

            res['epoch'].append(epoch+1)
            y_std = np.array(y_std, dtype=dtype_default)

            m, v = preds_f_mean_train, preds_f_var_train
            y_train_tiled = jnp.tile(y_train, (preds_f_samples_train.shape[0], 1, 1)).reshape(preds_f_samples_train.shape[0], -1)[:, :, None]
            res['pred_var_train'].append(v.mean())

            d = y_train_tiled - preds_f_samples_train
            # d = y_train - m
            du = d * y_std

            likelihood = tfd.Normal(preds_f_samples_train*y_std, noise_std*y_std)
            log_likelihood = dtype_default(jnp.mean(jnp.mean(likelihood.log_prob(y_train_tiled*y_std), 0), 0))
            # log_likelihood
            res['train_loglik'].append(log_likelihood)

            # ll = logsumexp(-0.5 * ((y_train_tiled - preds_f_samples_train) ** 2. / (noise_std) ** 2.), 0) - np.log(n_samples) - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(tau)
            # ll = norm.logpdf(y_train, loc=m, scale=v ** 0.5)
            # res['train_loglik'].append(np.mean(ll))

            res['train_rmse'].append(np.mean(np.mean(d ** 2, 1) ** 0.5))
            res['train_rmse_unnormalized'].append(np.mean(np.mean(du ** 2, 1) ** 0.5))

            m, v = preds_f_mean_test, preds_f_var_test
            y_test_tiled = jnp.tile(y_test, (preds_f_samples_test.shape[0], 1, 1)).reshape(preds_f_samples_test.shape[0], -1)[:, :, None]
            res['pred_var_test'].append(v.mean())

            d = y_test_tiled - preds_f_samples_test
            # d = y_test - m
            du = d * y_std

            likelihood = tfd.Normal(preds_f_samples_test*y_std, noise_std*y_std)
            log_likelihood = dtype_default(jnp.mean(jnp.mean(likelihood.log_prob(y_test_tiled*y_std), 0), 0))
            # log_likelihood
            res['test_loglik'].append(log_likelihood)

            # ll = logsumexp(-0.5 * ((y_test_tiled - preds_f_samples_test) ** 2. / (noise_std) ** 2.), 0) - np.log(n_samples) - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(tau)
            # ll = norm.logpdf(y_test, loc=m, scale=v ** 0.5)
            # res['test_loglik'].append(np.mean(ll))

            res['test_rmse'].append(np.mean(np.mean(d ** 2, 1) ** 0.5))
            res['test_rmse_unnormalized'].append(np.mean(np.mean(du ** 2, 1) ** 0.5))

            lu = norm.logpdf(y_test * y_std, loc=m * y_std, scale=(v ** 0.5) * y_std)
            res['test_loglik_unnormalized'].append(np.mean(np.mean(lu, 1)))

            res['test_mae'] = np.mean(np.mean(np.abs(d), 1))
            res['test_mae_unnormalized'].append(np.mean(np.mean(np.abs(du), 1)))

            elbo, log_likelihood, kl, scale = metrics._elbo_fsvi_regression(params, state, prior_mean, prior_cov, x_train, y_train, inducing_inputs, rng_key)
            res['elbo'].append(elbo)
            res['log_likelihood'].append(log_likelihood)
            res['kl'].appey_stdnd(kl)

            predicate = lambda module_name, name, value: name == 'w_mu' or name == 'b_mu'
            params_log_var = tree_flatten(hk.data_structures.partition(predicate, params)[1])[0]
            res['var_params'].append(np.exp(np.concatenate([np.array(params_log_var[i].flatten()) for i in range(len(params_log_var))], 0)))
            res['var_params_median'].append(np.median(np.exp(np.concatenate([np.array(params_log_var[i].flatten()) for i in range(len(params_log_var))], 0))))
            res['var_params_max'].append(np.max(np.exp(np.concatenate([np.array(params_log_var[i].flatten()) for i in range(len(params_log_var))], 0))))

            print(f"\nEpoch: {res['epoch'][-1]}\nTrain RMSE: {res['train_rmse_unnormalized'][-1]:.2f}  |  Train Standardized RMSE: {res['train_rmse'][-1]:.2f}  |  Train Log-Likelihood: {res['train_loglik'][-1]:.2f}\nTest RMSE: {res['test_rmse_unnormalized'][-1]:.2f}  |  Test Standardized RMSE: {res['test_rmse'][-1]:.2f}  |  Test Log-Likelihood: {res['test_loglik'][-1]:.2f}")
            print(f"ELBO: {res['elbo'][-1]:.0f}  |  Log-Likelihood: {res['log_likelihood'][-1]:.0f}  |  KL: {res['kl'][-1]:.0f}")

            if save:
                file_name = f'{save_path}/metrics.csv'
                with open(file_name, 'a') as metrics_file:
                    metrics_header = [
                        'Epoch',
                        'Train RMSE',
                        'Test RMSE',
                        'Train RMSE Standardized',
                        'Test RMSE Standardized',
                        'Train Log-Likelihood',
                        'Test Log-Likelihood',
                        'ELBO',
                        'Expected Log-Likelihood',
                        'KL',
                        'Train Mean Predictive Variance',
                        'Test Mean Predictive Variance'
                    ]
                    writer = csv.DictWriter(metrics_file, fieldnames=metrics_header)
                    if os.stat(file_name).st_size == 0:
                        writer.writeheader()
                    writer.writerow({
                        'Epoch': epoch+1,
                        'Train RMSE': res['train_rmse_unnormalized'][-1],
                        'Test RMSE': res['test_rmse_unnormalized'][-1],
                        'Train RMSE Standardized': res['train_rmse'][-1],
                        'Test RMSE Standardized': res['test_rmse'][-1],
                        'Train Log-Likelihood': res['train_loglik'][-1],
                        'Test Log-Likelihood': res['test_loglik'][-1],
                        'ELBO': res['elbo'][-1],
                        'Expected Log-Likelihood': res['log_likelihood'][-1],
                        'KL': res['kl'][-1],
                        'Train Mean Predictive Variance': res['pred_var_train'][-1],
                        'Test Mean Predictive Variance': res['pred_var_test'][-1]
                    })
                    metrics_file.close()

        if (epoch+1) % int(logging_frequency * 10) == 0 and epoch+1 >= 10000:
            fig, axs = plt.subplots(3, 2, figsize=(12,12))

            axs[0, 0].plot(res['epoch'], res['train_rmse_unnormalized'])
            axs[0, 0].plot(res['epoch'], res['test_rmse_unnormalized'])
            axs[0, 0].set_title('RMSE (not standardized)')

            axs[1, 0].plot(res['epoch'], res['train_loglik'])
            axs[1, 0].plot(res['epoch'], res['test_loglik'])
            axs[1, 0].set_title('Log-Likelihood')

            axs[2, 0].plot(res['epoch'], res['pred_var_train'])
            axs[2, 0].plot(res['epoch'], res['pred_var_test'])
            axs[2, 0].set_title('Mean Predictve Variance')

            ax_twin = axs[2, 0].twinx()
            ax_twin.plot(res['epoch'], res['var_params_median'], c='black')

            ax_twin.fill_between(
                res['epoch'],
                # np.array(res['var_params_median']) - 2 * np.array(res['var_params_std']),
                np.array(res['var_params_max']),
                color="C0",
                alpha=0.2,
            )

            axs[0, 1].plot(res['epoch'], res['elbo'])
            axs[0, 1].set_title(f"ELBO: {res['elbo'][-1]:.0f}")

            axs[1, 1].plot(res['epoch'], res['log_likelihood'])
            axs[1, 1].set_title(f"Expected Log-Likelihood: {res['log_likelihood'][-1]:.0f}")

            axs[2, 1].plot(res['epoch'], res['kl'])
            axs[2, 1].set_title(f"KL: {res['kl'][-1]:.0f}")

            if (epoch + 1) == epochs:
                fig.savefig(f'{save_path}/figures/training.pdf', bbox_inches='tight')

            plt.show()

            zoom = False
            if zoom:
                plot_range = int(res['epoch'][-1] * 0.8 / logging_frequency)

                fig, axs = plt.subplots(3, 2, figsize=(12, 12))

                axs[0, 0].plot(res['epoch'][-plot_range:], res['train_rmse_unnormalized'][-plot_range:])
                axs[0, 0].plot(res['epoch'][-plot_range:], res['test_rmse_unnormalized'][-plot_range:])
                axs[0, 0].set_title('Standardized RMSE')

                axs[1, 0].plot(res['epoch'][-plot_range:], res['train_loglik'][-plot_range:])
                axs[1, 0].plot(res['epoch'][-plot_range:], res['test_loglik'][-plot_range:])
                axs[1, 0].set_title('Log-Likelihood')

                axs[2, 0].plot(res['epoch'][-plot_range:], res['pred_var_train'][-plot_range:])
                axs[2, 0].plot(res['epoch'][-plot_range:], res['pred_var_test'][-plot_range:])
                axs[2, 0].set_title('Mean Predictve Variance')

                axs[0, 1].plot(res['epoch'][-plot_range:], res['elbo'][-plot_range:])
                axs[0, 1].set_title(f"ELBO: {res['elbo'][-1]:.0f}")

                axs[1, 1].plot(res['epoch'][-plot_range:], res['log_likelihood'][-plot_range:])
                axs[1, 1].set_title(f"Expected Log-Likelihood: {res['log_likelihood'][-1]:.0f}")

                axs[2, 1].plot(res['epoch'][-plot_range:], res['kl'][-plot_range:])
                axs[2, 1].set_title(f"KL: {res['kl'][-1]:.0f}")

                plt.show()

        if (epoch + 1) % int(logging_frequency * 100) == 0 and epoch > 0:

            # for i in range(len(res['var_params'])):
            #     plt.hist(res['var_params'][i], bins=100, alpha=1 / ((len(res['var_params'])-i)))
            # plt.show()

            xx = np.array([np.array(preds_f_samples_train[:, id, 0] * y_std).tolist() for id in ids_train]).flatten()
            # xx += np.array([np.repeat(i, n_samples) * 2 for i in range(num_eval_points)]).flatten() - num_eval_points
            yy = np.array([np.repeat(i, n_samples) for i in range(num_eval_points)]).flatten()

            df = pd.DataFrame()
            df['samples'] = pd.Series(xx)
            df['input'] = pd.Series(yy)

            fig_histogram, axes_histogram = joypy.joyplot(df, by='input', column='samples', grid="y", linewidth=1, legend=False,
                                      alpha=0.4, fade=False, figsize=(8, 5), kind="kde", bins=100, tails=0.2, range_style='own',
                                                          title='Predictive Marginal Distribution on Training Inputs')
            # plt.show()
            i = 0
            for a in axes_histogram[0:-1]:
                a.axvline(x=y_train[ids_train[i]] * y_std, c='black', ymin=0.15, ymax=0.25)
                # a.axvline(x=y_test[ids_train[i]] + 2 * i - num_eval_points, c='black')
                i += 1
                # a.set_xlim(-2.5, 2.5)
                # a.set_xlim([num_eval_points - 3, num_eval_points + 3])
                xlims = a.get_xlim()

            if (epoch + 1) == epochs:
                fig_histogram.savefig(f'{save_path}/figures/training_marginals.pdf', bbox_inches='tight')

            plt.show()

            params_mean, params_log_var, params_deterministic = partition_params(params)

            gaussian_mean, gaussian_cov = utils_linearization.bnn_linearized_predictive(apply_fn, params_mean,
                                                                       params_log_var, params_deterministic, state, x_train[ids_train, :],
                                                                       rng_key)
            gaussian_samples = []
            for i in range(n_samples):
                gaussian_samples.append(np.random.normal(gaussian_mean.squeeze(), np.sqrt(np.diag(gaussian_cov.squeeze()))))
            gaussian_samples = np.array(gaussian_samples)

            xx = np.array([np.array(gaussian_samples[:, id] * y_std).tolist() for id in range(ids_train.shape[0])]).flatten()
            # xx += np.array([np.repeat(i, n_samples) * 2 for i in range(num_eval_points)]).flatten() - num_eval_points
            yy = np.array([np.repeat(i, n_samples) for i in range(num_eval_points)]).flatten()

            df = pd.DataFrame()
            df['samples'] = pd.Series(xx)
            df['input'] = pd.Series(yy)

            fig_histogram, axes_histogram = joypy.joyplot(df, by='input', column='samples', grid="y", linewidth=1, legend=False,
                                      alpha=0.4, fade=False, figsize=(8, 5), kind="kde", bins=100, tails=0.2, range_style='own',
                                                          title='Gaussian Predictive Marginal Distribution on Training Inputs')
            plt.show()
            i = 0
            for a in axes_histogram[0:-1]:
                a.axvline(x=y_train[ids_train[i]] * y_std, c='black', ymin=0.15, ymax=0.25)
                # a.axvline(x=y_test[ids_train[i]] + 2 * i - num_eval_points, c='black')
                i += 1
            for a in axes_histogram:
                # a.set_xlim(-2.5, 2.5)
                # a.set_xlim([num_eval_points - 3, num_eval_points + 3])
                # a.set_xlim(xlims)
                a.set_xlim(xlims)

            if (epoch + 1) == epochs:
                fig_histogram.savefig(f'{save_path}/figures/training_marginals_gaussian.pdf', bbox_inches='tight')

            plt.show()

            xx = np.array([np.array(preds_f_samples_test[:, id, 0] * y_std).tolist() for id in ids_test]).flatten()
            # xx += np.array([np.repeat(i, n_samples) * 2 for i in range(num_eval_points)]).flatten() - num_eval_points
            yy = np.array([np.repeat(i, n_samples) for i in range(num_eval_points)]).flatten()

            df = pd.DataFrame()
            df['samples'] = pd.Series(xx)
            df['input'] = pd.Series(yy)

            fig_histogram, axes_histogram = joypy.joyplot(df, by='input', column='samples', grid="y", linewidth=1, legend=False,
                                      alpha=0.4, fade=False, figsize=(8, 5), kind="kde", bins=1000, tails=0.2, range_style='own',
                                                          title='Predictive Marginal Distribution on Test Inputs')
            # plt.show()
            i = 0
            for a in axes_histogram[0:-1]:
                a.axvline(x=y_test[ids_test[i]] * y_std, c='black', ymin=0.15, ymax=0.25)
                # a.axvline(x=y_test[ids_test[i]] + 2 * i - num_eval_points, c='black')
                i += 1
                # a.set_xlim(-2.5, 2.5)
                # a.set_xlim([num_eval_points - 3, num_eval_points + 3])

            if (epoch + 1) == epochs:
                fig_histogram.savefig(f'{save_path}/figures/test_marginals.pdf', bbox_inches='tight')

            plt.show()






        # preds_f_samples_test_ = preds_f_samples_test * y_std + y_mean
        # y_test_tiled_ = y_test_tiled * y_std + y_mean
        #
        # np.mean(logsumexp(-0.5 * tau * ((y_test_tiled - preds_f_samples_test)) ** 2., 0) - np.log(n_samples) - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(tau))
        # np.mean(logsumexp(-0.5 * ((y_test_tiled_ - preds_f_samples_test_) ** 2. / (noise_std * y_std) ** 2. ) , 0) - np.log(n_samples) - 0.5 * np.log(2 * np.pi) - 0.5 * np.log((noise_std) ** 2.))
        #
        # rng_key, _ = jax.random.split(rng_key)
        # from tensorflow_probability.substrates import jax as tfp
        # tfd = tfp.distributions
        # y_std_ = np.array(y_std, dtype=dtype_default)
        # likelihood = tfd.Normal(preds_f_samples_test[:, :, 0], noise_std/10)
        # log_likelihood = jnp.sum(jnp.mean(likelihood.log_prob(y_test_tiled[:, :, 0]), 0), 0)
        # log_likelihood / y_test_tiled.shape[1]






    #%%

    ####### Saving model parameters #######
    if save:
        with open(f'saved_models/uci/{dataset}/params_pickle_{seed}', 'wb') as file:
            pickle.dump(params, file)


def uci_mfvi_mlp(
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
                    debug
):
    rng_key = random.PRNGKey(seed)
    rng_key, _ = random.split(rng_key)

# >>>>>>>>>>>>>>>>>>> Model-specific setup begins below

    data = get_regression_data('energy', split=seed)
    x_train, y_train, x_test, y_test = data.X_train, data.Y_train, data.X_test, data.Y_test
    noise_std = 1.0

    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]

    # x_train = x_train.reshape([-1, 1])
    y_train = y_train.reshape([-1, 1])
    # x_test = x_test.reshape([-1, 1])

    n_train = x_train.shape[0]
    batch_size = n_train
    n_batches = np.int(n_train / batch_size)

    if architecture != []:
        for i in range(0, len(architecture)):
            architecture[i] = int(architecture[i])

    stochastic_parameters = True

    # Initialize NN
    model = MLP(output_dim=output_dim,
                activation_fn=activation,
                architecture=architecture,
                stochastic_parameters=stochastic_parameters)
    metrics = Objectives(
                         model=model,
                         predict_f=model.predict_f,
                         predict_y=model.predict_y,
                         predict_y_multisample=model.predict_y_multisample,
                         output_dim=output_dim,
                         kl_scale=kl_scale,
                         full_cov=full_cov,
                         noise_std=noise_std,
                         n_batches=n_batches)
    loss = metrics.nelbo_mfvi_regression
    accuracy = metrics.gaussian_log_likelihood

    # Initialize NN
    opt = optax.adam(learning_rate)
    x_batch = x_train[0,:]
    params_init, state = model.initialize(rng_key, x_batch)
    opt_state = opt.init(params_init)
    params = params_init

    @jit
    def update(params, state, opt_state, prior_mean, prior_cov, x_batch, y_batch, inducing_inputs, rng):
        rng_key, _ = random.split(rng)
        params_copy = params
        grads = jax.grad(loss, argnums = 0)(params, params_copy, state, prior_mean, prior_cov, x_batch, y_batch, inducing_inputs, rng_key)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    print(f'\n--- Training for {epochs} epochs ---\n')
    for epoch in tqdm(range(epochs)):
        for batch_idx in range(n_batches):
            rng_key, subkey = random.split(rng_key)
            x_batch = x_train #[batch_idx, :]
            y_batch = y_train #[batch_idx, :]
            params, opt_state = update(params, state, opt_state, prior_mean, prior_cov, x_batch, y_batch, None, rng_key)

#%%

# >>>>>>>>>>>>>>>>>>> Model-specific setup ends here

        if epoch % logging_frequency == 0:
            predicted_label_samples_train, predicted_labels_mean_train, predicted_labels_var_train = model.predict_f_multisample(params, state, x_train, rng_key, 50)
            predicted_label_samples_test, predicted_labels_mean_test, predicted_labels_var_test = model.predict_f_multisample(params, state, x_test, rng_key, 50)
            # noise_var = 0.

            res = {}

            m, v = predicted_labels_mean_train, predicted_labels_var_train
            d = data.Y_train - m
            du = d * data.Y_std

            l = norm.logpdf(data.Y_train, loc=m, scale=v ** 0.5)
            res['train_loglik'] = np.average(l)

            res['train_rmse'] = np.average(d ** 2) ** 0.5
            res['train_rmse_unnormalized'] = np.average(du ** 2) ** 0.5

            m, v = predicted_labels_mean_test, predicted_labels_var_test
            d = data.Y_test - m
            du = d * data.Y_std

            l = norm.logpdf(data.Y_test, loc=m, scale=v ** 0.5)
            res['test_loglik'] = np.average(l)

            lu = norm.logpdf(data.Y_test * data.Y_std, loc=m * data.Y_std, scale=(v ** 0.5) * data.Y_std)
            res['test_loglik_unnormalized'] = np.average(lu)

            res['test_mae'] = np.average(np.abs(d))
            res['test_mae_unnormalized'] = np.average(np.abs(du))

            res['test_rmse'] = np.average(d ** 2) ** 0.5
            res['test_rmse_unnormalized'] = np.average(du ** 2) ** 0.5

            print(f"\nEpoch: {epoch + 1}\nTrain Standardized RMSE: {res['train_rmse']:.2f}  |  Train Log-Likelihood: {res['train_loglik']:.2f}\nTest Standardized RMSE: {res['test_rmse']:.2f}  |  Test Log-Likelihood: {res['test_loglik']:.2f}")


#%%

    ####### Saving model parameters #######
    if save:
        with open(f'saved_models/uci/params_pickle', 'wb') as file:
            pickle.dump(params, file)
