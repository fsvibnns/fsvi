import os
from tqdm import tqdm
import time
import csv
import pickle
import getpass
from scipy.stats import norm
from jax.scipy.special import logsumexp
import seqtools

import jax
import jax.numpy as jnp
from jax import jit, grad, random
from jax.tree_util import tree_flatten
import haiku as hk

import optax
import numpy as np
from sklearn.decomposition import PCA

import pandas as pd
try:
    import joypy
except:
    pass

from gym.envs.mujoco import HalfCheetahEnv, HopperEnv, Walker2dEnv, AntEnv
import gym
import mj_envs

import random as random_py
import datetime
from scipy.stats import norm

from fsvi.utils.utils_training import Training
from fsvi.utils import datasets
from fsvi.utils.utils import get_minibatch, initialize_random_keys

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

from fsvi.utils import utils
from fsvi.utils import utils_logging
from fsvi.utils import utils_training
from fsvi.models.networks import MLP as MLP, ACTIVATION_DICT

dtype_default = jnp.float32
# np.set_printoptions(suppress=True)
eps = 1e-6

# name = 'door'
# env = gym.make('{}-binary-v0'.format(name))
from mjrl.utils.gym_env import GymEnv
env = GymEnv('door-binary-v0')

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


def rollout(env, model, max_path_length=np.inf, double=False, render=False):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next element will be a list of dictionaries, with the index into
    the list being the index into the time
     - env_infos
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0

    while path_length < max_path_length:
        # Can pull log probs from here, numpy -> torch -> numpy

        # start = datetime.datetime.now()

        o_input = np.array([o])
        preds_f_mean = model(o_input)
        a = preds_f_mean

        if len(a) == 1:
            a = a[0]

        # finish = datetime.datetime.now()
        #
        # print(finish - start)
        # print(a)

        if render:
            env.render()

        next_o, r, d, env_info = env.step(a)

        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack((observations[1:, :], np.expand_dims(next_o, 0)))
    return dict(observations=observations, actions=actions, rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations, terminals=np.array(terminals).reshape(-1, 1), env_infos=env_infos, )


def collect_new_paths(env, model, max_path_length, num_steps, discard_incomplete_paths, double=False, render=False):
    paths = []
    num_steps_collected = 0
    it = 0
    while num_steps_collected < num_steps:
        max_path_length_this_loop = min(  # Do not go over num_steps
            max_path_length, num_steps - num_steps_collected, )

        path = rollout(env, model, max_path_length=max_path_length_this_loop, double=double, render=render)

        path_len = len(path['actions'])
        if (path_len != max_path_length and not path['terminals'][-1] and discard_incomplete_paths):
            break
        num_steps_collected += path_len
        paths.append(path)
        it += 1
        print(f"Completed Rollouts: {it}/{int(num_steps/max_path_length)}")
    return paths


def offline_rl_door_eval(
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

    input_dim = 39
    output_dim = 28

    max_path_length = 200
    n_test_episodes = 10
    render = True

    if getpass.getuser() == 'ANON':
        path = '/scratch-ssd/ANON/deployment/testing/projects/function-space-variational-inference/saved_models/offline_rl/'
    else:
        path = '/Volumes/Data/Google_Drive/AYA_Google_Drive/Code/projects/function-space-variational-inference/saved_models/offline_rl/'

    # architecture = [256, 512, 256, 128]
    # params = np.load(path + 'params_pickle', allow_pickle=True)

    architecture = ''
    for i in kwargs['architecture']:
        architecture = architecture + str(i) + '_'
    architecture = architecture[:-1]

    params = np.load(path + f'params_pickle_{architecture}', allow_pickle=True)

    # architecture = [256, 512, 256, 128]
    # params = np.load(path + 'params_pickle_01_256_512_256_128', allow_pickle=True)
    # params = np.load(path + 'params_pickle_02_256_512_256_128', allow_pickle=True)
    # params = np.load(path + 'params_pickle_05_256_512_256_128', allow_pickle=True)  # 1000 rollouts  |  Epoch eval, Eval Reward: -199.196, Std: 3.17  |  Success Rate: 8.00%
    # params = np.load(path + 'params_pickle_06_256_512_256_128', allow_pickle=True)  # 1000 rollouts  |  Epoch eval, Eval Reward: -198.376, Std: 4.34  |  Success Rate: 16.00%
    # params = np.load(path + 'params_pickle_07_256_512_256_128', allow_pickle=True)  # 1000 rollouts  |  Epoch eval, Eval Reward: -197.968, Std: 5.29  |  Success Rate: 17.00%
    # params = np.load(path + 'params_pickle_08_256_512_256_128', allow_pickle=True)  # 1000 rollouts  |  Epoch eval, Eval Reward: -197.925, Std: 6.27  |  Success Rate: 14.00%
    # params = np.load(path + 'params_pickle_09_256_512_256_128', allow_pickle=True)  # 1000 rollouts  |  Epoch eval, Eval Reward: -196.258, Std: 9.59  |  Success Rate: 17.00%

    # architecture = [512, 512, 512, 256]
    # params = np.load(path + 'params_pickle_01_256_512_256_128', allow_pickle=True)

    training = Training(
        input_shape=[10,input_dim],
        output_dim=output_dim,
        n_train=10,
        batch_size=1,
        n_batches=1,
        full_ntk=False,
        model_type=model_type,
        **kwargs,
    )

    # INITIALIZE MODEL
    model, init_fn, apply_fn, state, params = training.initialize_model(
        rng_key=rng_key,
    )

    model.stochastic_parameters = False
    pred_fn = lambda x: model.predict_f(params, state, x, rng_key, False)
    # pred_fn = lambda x: model.predict_f_multisample(params, state, x, rng_key, n_samples=100)
    # preds_f_samples, preds_f_mean, preds_f_var = pred_fn(o)

    # Generate rollout from the policy
    start = datetime.datetime.now()

    ps = collect_new_paths(env, pred_fn, max_path_length, max_path_length * n_test_episodes,
                           discard_incomplete_paths=True, render=render)

    finish = datetime.datetime.now()
    print("Profiling took: ", finish - start)

    eval_rew = np.mean([np.sum(p['rewards']) for p in ps])
    eval_std = np.std([np.sum(p['rewards']) for p in ps])
    print('Epoch {}, Eval Reward: {}, Std: {}'.format('eval', eval_rew, eval_std))

    rews = [np.sum(p['rewards']) for p in ps]
    blist = [1 if k > -max_path_length else 0 for k in rews]
    perc = int(dtype_default(sum(blist)) / dtype_default(len(blist)) * 100)

    print(f"Success Rate: {perc:.2f}%")

    print('done')

    # TODO: set
    # if saving_policies:
    #     print('Saved')
    #     torch.save(model.state_dict(),
    #                'offline_rl/{}/_{}_{}_{}_{}_{}p.pt'.format(args.save_dir, name, args.gp_type, epoch,
    #                                                       int(eval_rew), perc))


def offline_rl(
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

    if getpass.getuser() == 'ANON':
        path = '/scratch-ssd/ANON/deployment/testing/data/large/offpolicy_hand_data/'
    if getpass.getuser() == 'ANON':
        path = '/Volumes/Data/Google_Drive/AYA_Google_Drive/Code/data/large/offpolicy_hand_data/'

    # data = np.load(path+'door-v0_demos.pickle', allow_pickle=True)
    # X = np.array([j for i in [traj['observations'] for traj in data] for j in i], dtype=dtype_default)
    # Y = np.array([j for i in [traj['actions'] for traj in data] for j in i], dtype=dtype_default)

    data = np.load('/scratch-ssd/ANON/deployment/testing/data/large/offpolicy_hand_data/door2_sparse.npy', allow_pickle=True)
    if type(data[0]['observations'][0]) is dict:
        # Convert to just the states
        for traj in data:
            traj['observations'] = [t['state_observation'] for t in traj['observations']]
    X = np.array([j for i in [traj['observations'] for traj in data] for j in i], dtype=dtype_default)
    Y = np.array([j for i in [traj['actions'] for traj in data] for j in i], dtype=dtype_default)

    # data = np.load('/scratch-ssd/ANON/deployment/testing/data/large/offpolicy_hand_data/door-v0_demos.pickle', allow_pickle=True)
    # X = np.array([j for i in [traj['observations'][:150] for traj in data] for j in i])
    # Y = np.array([j for i in [traj['actions'][:150] for traj in data] for j in i])

    n = X.shape[0]
    if batch_size == 0:
        batch_size = n

    val_data = 'matern'
    if val_data == 'expert':
        x_mean = X.mean(0)
        x_std = X.std(0)
        y_mean = Y.mean(0)
        y_std = Y.std(0)

        X = (X - x_mean) / x_std
        Y = (Y - y_mean) / y_std

        train_test_split_ratio = 0.8
        n = X.shape[0]
        n_train = int(n * train_test_split_ratio)
        # n_test = n - n_train

        idx = np.random.permutation(n)
        idx_train = idx[0:n_train]
        idx_test = idx[n_train:]

        x_train = X[idx_train]
        x_test = X[idx_test]
        y_train = Y[idx_train]
        y_test = Y[idx_test]

    if val_data == 'matern':
        x_train = np.array(X, dtype=dtype_default)
        y_train = np.array(Y, dtype=dtype_default)
        # n_train = n

        if getpass.getuser() == 'ANON':
            path = '/scratch-ssd/ANON/deployment/testing/data/large/'
        if getpass.getuser() == 'ANON':
            path = '/Volumes/Data/Google_Drive/AYA_Google_Drive/Code/data/large/offpolicy_hand_data/'

        data = np.load(path + 'matern_door_validation.pkl', allow_pickle=True)
        X = np.array([j for i in [traj['observations'] for traj in data] for j in i], dtype=dtype_default)
        Y = np.array([j for i in [traj['actions'] for traj in data] for j in i], dtype=dtype_default)

        x_test = X
        y_test = Y
        # n_test = x_test.shape[0]

        X = np.concatenate([x_train, x_test], 0)
        Y = np.concatenate([y_train, y_test], 0)

        # x_mean = X.mean(0)
        # x_std = X.std(0)
        y_mean = Y.mean(0)
        y_std = Y.std(0)

        # x_train = (x_train - x_mean) / x_std
        # x_test = (x_test - x_mean) / x_std

        # y_train = (y_train - y_mean) / y_std
        # y_test = (y_test - y_mean) / y_std

    kwargs['noise_std'] = noise_std = 0.001
    # noise_std = 1.0
    noise_var = noise_std ** 2
    kwargs['tau'] = tau = 1 / noise_var

    input_dim = x_train.shape[1]
    input_shape = list(x_train.shape)
    output_dim = y_train.shape[1]

    n_train = x_train.shape[0]
    if batch_size < n:
        n_batches = np.int(n_train / batch_size) + 1
    else:
        n_batches = np.int(n_train / batch_size)
    num_eval_points = 20
    ids = np.random.choice(y_train.shape[1], num_eval_points, replace=False)
    # ids = range(num_eval_points)
    ids_train = ids
    ids_test = ids
    # ids_train = ids[np.argsort(y_train[ids][:,0])]
    # ids_test = ids[np.argsort(y_test[ids][:,0])]

    # DEFINE NUMPY TRAINLOADER  # TODO: test implementation
    n_train = x_train.shape[0]
    train_dataset = seqtools.collate([x_train, y_train])
    if batch_size == 0:
        trainloader_ = seqtools.batch(train_dataset, n_train, collate_fn=datasets.collate_fn)
    else:
        trainloader_ = seqtools.batch(train_dataset, batch_size, collate_fn=datasets.collate_fn)

    trainloader = []
    for i, data in enumerate(trainloader_, 0):
        x_batch = jnp.array(data[0], dtype=dtype_default)
        y_batch = jnp.array(data[1], dtype=dtype_default)
        trainloader.append([x_batch, y_batch])

    zoom = False
    evaluate = True
    plot_marginals = False
    plot_histogram = False
    plot_uncertainty = False
    render = False

    # INITIALIZE TRAINING CLASS
    training = Training(
        input_shape=input_shape,
        output_dim=output_dim,
        n_train=n_train,
        batch_size=batch_size,
        n_batches=n_train // batch_size,
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
    val_frac = 0.0
    x_train_permuted = x_train
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
        y_test=y_test,
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


    print(f'\n--- Training for {epochs} epochs ---\n')
    for epoch in range(epochs):
        logging.t0 = time.time()

        for i, data in enumerate(trainloader, 0):
            rng_key_train, _ = random.split(rng_key_train)

            x_batch, y_batch = get_minibatch(
                data, output_dim, input_shape, prediction_type
            )
            inducing_inputs = inducing_input_fn(x_batch, rng_key_train)

            # TODO: clean up
            if 'mlp' in model_type:
                # x_batch = x_batch.reshape(batch_size, -1)
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

#%%

        if epoch % kwargs['logging_frequency'] == 0:
            n_samples = 50
        else:
            n_samples = 10
        preds_f_samples_train, preds_f_mean_train, preds_f_var_train = model.predict_f_multisample(params, state, x_train.squeeze(), rng_key, n_samples, is_training=False)
        preds_f_samples_test, preds_f_mean_test, preds_f_var_test = model.predict_f_multisample(params, state, x_test, rng_key, n_samples, is_training=False)

        res['epoch'].append(epoch+1)

        m, v = preds_f_mean_train, preds_f_var_train
        y_train_tiled = jnp.tile(y_train, (preds_f_samples_train.shape[0], 1, 1)).reshape(preds_f_samples_train.shape[0], y_train.shape[0], y_train.shape[1])
        res['pred_var_train'].append(v.mean())

        d = y_train_tiled - preds_f_samples_train
        # d = y_train - m
        # du = d * y_std
        du = d
        # du = d * y_std

        ll = logsumexp(-0.5 * tau * ((y_train_tiled - preds_f_samples_train) * y_std) ** 2., 0) - np.log(n_samples) - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(tau)
        # ll = norm.logpdf(y_train, loc=m, scale=v ** 0.5)
        res['train_loglik'].append(np.mean(ll))

        res['train_rmse'].append(np.mean(np.mean(d ** 2, 1) ** 0.5))
        res['train_rmse_unnormalized'].append(np.mean(np.mean(du ** 2, 1) ** 0.5))

        m, v = preds_f_mean_test, preds_f_var_test
        y_test_tiled = jnp.tile(y_test, (preds_f_samples_test.shape[0], 1, 1)).reshape(preds_f_samples_test.shape[0], y_test.shape[0], y_test.shape[1])
        res['pred_var_test'].append(v.mean())

        d = y_test_tiled - preds_f_samples_test
        # d = y_test - m
        du = d
        # du = d * y_std

        ll = logsumexp(-0.5 * tau * ((y_test_tiled - preds_f_samples_test)) ** 2., 0) - np.log(n_samples) - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(tau)
        # ll = logsumexp(-0.5 * tau * ((y_test_tiled - preds_f_samples_test) * y_std) ** 2., 0) - np.log(n_samples) - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(tau)
        # ll = norm.logpdf(y_test, loc=m, scale=v ** 0.5)
        res['test_loglik'].append(np.mean(ll))

        res['test_rmse'].append(np.mean(np.mean(d ** 2, 1) ** 0.5))
        res['test_rmse_unnormalized'].append(np.mean(np.mean(du ** 2, 1) ** 0.5))

        lu = norm.logpdf(y_test, loc=m, scale=(v ** 0.5))
        # lu = norm.logpdf(y_test * y_std, loc=m * y_std, scale=(v ** 0.5) * y_std)
        res['test_loglik_unnormalized'].append(np.mean(np.mean(lu, 1)))

        res['test_mae'] = np.mean(np.mean(np.abs(d), 1))
        res['test_mae_unnormalized'].append(np.mean(np.mean(np.abs(du), 1)))

        # elbo, log_likelihood, kl, scale = metrics._elbo_fsvi_regression(params, state, prior_mean, prior_cov, x_train, y_train, inducing_inputs, rng_key, is_training=True)
        # res['elbo'].append(elbo)
        # res['log_likelihood'].append(log_likelihood)
        # res['kl'].append(kl)

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch: {epoch+1}\nTrain RMSE: {res['train_rmse_unnormalized'][-1]:.3f}  |  Train Standardized RMSE: {res['train_rmse'][-1]:.3f}  |  Train Log-Likelihood: {res['train_loglik'][-1]:.3f}\nTest RMSE: {res['test_rmse_unnormalized'][-1]:.3f}  |  Test Standardized RMSE: {res['test_rmse'][-1]:.3f}  |  Test Log-Likelihood: {res['test_loglik'][-1]:.3f}")
            # print(f"ELBO: {res['elbo'][-1]:.0f}   |  Log-Likelihood: {res['log_likelihood'][-1]:.0f}   |  KL: {res['kl'][-1]:.0f}")

        if epoch % (kwargs['logging_frequency'] + 1) == 0 and epoch > 0:
            with open(f'saved_models/offline_rl/params_pickle', 'wb') as file:
                pickle.dump(params, file)

            fig, axs = plt.subplots(3, 2, figsize=(12,12))

            axs[0, 0].plot(res['epoch'], res['train_rmse'])
            axs[0, 0].plot(res['epoch'], res['test_rmse'])
            axs[0, 0].set_title('Standardized RMSE')

            axs[1, 0].plot(res['epoch'], res['train_loglik'])
            axs[1, 0].plot(res['epoch'], res['test_loglik'])
            axs[1, 0].set_title('Log-Likelihood')

            axs[2, 0].plot(res['epoch'], res['pred_var_train'])
            axs[2, 0].plot(res['epoch'], res['pred_var_test'])
            axs[2, 0].set_title('Mean Predictve Variance')

            # ax_twin = axs[2, 0].twinx()
            # ax_twin.plot(res['epoch'], res['var_params_median'], c='black')
            #
            # ax_twin.fill_between(
            #     res['epoch'],
            #     # np.array(res['var_params_median']) - 2 * np.array(res['var_params_std']),
            #     np.array(res['var_params_max']),
            #     color="C0",
            #     alpha=0.2,
            # )
            #
            # axs[0, 1].plot(res['epoch'], res['elbo'])
            # axs[0, 1].set_title(f"ELBO: {res['elbo'][-1]:.0f}")
            #
            # axs[1, 1].plot(res['epoch'], res['log_likelihood'])
            # axs[1, 1].set_title(f"Expected Log-Likelihood: {res['log_likelihood'][-1]:.0f}")
            #
            # axs[2, 1].plot(res['epoch'], res['kl'])
            # axs[2, 1].set_title(f"KL: {res['kl'][-1]:.0f}")

            plt.show()

            if zoom:
                plot_range = int(epoch * 0.8 / kwargs['logging_frequency'])

                fig, axs = plt.subplots(3, 2, figsize=(12, 12))

                axs[0, 0].plot(res['epoch'][-plot_range:], res['train_rmse'][-plot_range:])
                axs[0, 0].plot(res['epoch'][-plot_range:], res['test_rmse'][-plot_range:])
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

            if plot_histogram:
                predicate = lambda module_name, name, value: name == 'w_mu' or name == 'b_mu'
                params_log_var = tree_flatten(hk.data_structures.partition(predicate, params)[1])[0]
                res['var_params'].append(np.exp(np.concatenate([np.array(params_log_var[i].flatten()) for i in range(len(params_log_var))], 0)))
                res['var_params_median'].append(np.median(np.exp(np.concatenate([np.array(params_log_var[i].flatten()) for i in range(len(params_log_var))], 0))))
                res['var_params_max'].append(np.max(np.exp(np.concatenate([np.array(params_log_var[i].flatten()) for i in range(len(params_log_var))], 0))))

                for i in range(len(res['var_params'])):
                    plt.hist(res['var_params'][i], bins=100, alpha=1 / ((len(res['var_params'])-i)))
                plt.show()

            if plot_uncertainty:
                data = np.load('/scratch-ssd/ANON/deployment/testing/data/large/offpolicy_hand_data/door-v0_demos.pickle', allow_pickle=True)
                x_pca = np.array([j for i in [traj['observations'][:150] for traj in data] for j in i])

                pca = PCA(n_components=2)
                pca.fit(x_pca)
                pca_res = pca.transform(x_pca)

                points = 100
                pca_grid_points = np.linspace(-10, 10, points)

                pca_xx, pca_yy = np.meshgrid(pca_grid_points, pca_grid_points, indexing='ij')

                pca_xx = np.reshape(pca_xx, (points * points, 1))
                pca_yy = np.reshape(pca_yy, (points * points, 1))

                pca_coords = np.concatenate([pca_xx, pca_yy], axis=1)

                obs_pca = pca.inverse_transform(pca_coords)

                _, _, preds_f_var_pca = model.predict_f_multisample(params,
                                                                    state,
                                                                    obs_pca,
                                                                    rng_key,
                                                                    n_samples,
                                                                    is_training=False)

                preds_f_var_pca = preds_f_var_pca
                preds_f_var_pca = np.reshape(preds_f_var_pca.mean(-1), [100, 100])

                pca_xx, pca_yy = np.meshgrid(pca_grid_points, pca_grid_points, indexing='ij')

                levels = 10
                levels = np.linspace(0, 0.01, levels)
                color = 'black'
                contours = True

                fig = plt.figure(figsize=(6, 4.5))
                ax = fig.add_subplot(111)

                # cpf = plt.contourf(pca_xx, pca_yy, std_nn_,levels=levels, cmap='coolwarm')
                cpf = plt.contourf(pca_xx, pca_yy, preds_f_var_pca, levels=levels, cmap='coolwarm')
                plt.colorbar()
                line_colors = ['black' for l in cpf.levels]
                if contours:
                    cp = ax.contour(pca_xx, pca_yy, preds_f_var_pca, levels=levels, colors=line_colors)
                # cp = ax.contour(pca_xx, pca_yy, std_gp_, levels=levels)

                ax.set_xlim(-3, 3)
                ax.set_ylim(-3, 3)

                for i in range(30):
                    plt.plot(pca_res[i * 150:(i + 1) * 150, 0], pca_res[i * 150:(i + 1) * 150, 1], zorder=100, color=color)

                plt.show()

                fig = plt.figure(figsize=(6, 4.5))
                ax = fig.add_subplot(111)

                # levels = np.linspace(0, 0.24, levels)
                # cpf = plt.contourf(pca_xx, pca_yy, std_nn_,levels=levels, cmap='coolwarm')
                cpf = plt.contourf(pca_xx, pca_yy, preds_f_var_pca, levels=levels, cmap='coolwarm')
                plt.colorbar()
                line_colors = ['black' for l in cpf.levels]
                if contours:
                    cp = ax.contour(pca_xx, pca_yy, preds_f_var_pca, levels=levels, colors=line_colors)
                # cp = ax.contour(pca_xx, pca_yy, std_gp_, levels=levels)

                ax.set_xlim(-8, 8)

                for i in range(30):
                    plt.plot(pca_res[i * 150:(i + 1) * 150, 0], pca_res[i * 150:(i + 1) * 150, 1], zorder=100, color=color)

                plt.show()

            if plot_marginals:
                for d in range(y_mean.shape[-1]):
                    xx = np.array([np.array(preds_f_samples_train[:, id, d] * y_std[d]).tolist() for id in ids_train]).flatten()
                    # xx += np.array([np.repeat(i, n_samples) * 2 for i in range(num_eval_points)]).flatten() - num_eval_points
                    yy = np.array([np.repeat(i, n_samples) for i in range(num_eval_points)]).flatten()

                    df = pd.DataFrame()
                    df['samples'] = pd.Series(xx)
                    df['input'] = pd.Series(yy)

                    fig_histogram, axes_histogram = joypy.joyplot(df, by='input', column='samples', grid="y", linewidth=1, legend=False,
                                              alpha=0.4, fade=False, figsize=(8, 5), kind="kde", bins=100, tails=0.2, range_style='own',
                                                                  title='Predictive Marginal Distribution on Training Inputs, Dimension={d+1}')
                    # plt.show()
                    i = 0
                    for a in axes_histogram[0:-1]:
                        a.axvline(x=y_train[0, ids_train[i], d] * y_std[d], c='black', ymin=0.15, ymax=0.25)
                        # a.axvline(x=y_test[ids_train[i]] + 2 * i - num_eval_points, c='black')
                        i += 1
                        # a.set_xlim(-2.5, 2.5)
                        # a.set_xlim([num_eval_points - 3, num_eval_points + 3])
                    plt.show()

                    xx = np.array([np.array(preds_f_samples_test[:, id, d] * y_std[d]).tolist() for id in ids_test]).flatten()
                    # xx += np.array([np.repeat(i, n_samples) * 2 for i in range(num_eval_points)]).flatten() - num_eval_points
                    yy = np.array([np.repeat(i, n_samples) for i in range(num_eval_points)]).flatten()

                    df = pd.DataFrame()
                    df['samples'] = pd.Series(xx)
                    df['input'] = pd.Series(yy)

                    fig_histogram, axes_histogram = joypy.joyplot(df, by='input', column='samples', grid="y", linewidth=1, legend=False,
                                              alpha=0.4, fade=False, figsize=(8, 5), kind="kde", bins=1000, tails=0.2, range_style='own',
                                                                  title='Predictive Marginal Distribution on Test Inputs, Dimension={d+1}')
                    # plt.show()
                    i = 0
                    for a in axes_histogram[0:-1]:
                        a.axvline(x=y_test[ids_test[i], d] * y_std[d], c='black', ymin=0.15, ymax=0.25)
                        # a.axvline(x=y_test[ids_test[i]] + 2 * i - num_eval_points, c='black')
                        i += 1
                        # a.set_xlim(-2.5, 2.5)
                        # a.set_xlim([num_eval_points - 3, num_eval_points + 3])
                    plt.show()

        if epoch ==0 or epoch % (kwargs['logging_frequency'] * 100) == 0 and epoch > 0 and evaluate:
            max_path_length = 200
            n_test_episodes = 30
            start = datetime.datetime.now()

            model.stochastic_parameters = False
            pred_fn = lambda x: model.predict_f(params, state, x, rng_key, False)
            ps = collect_new_paths(env, pred_fn, max_path_length, max_path_length * n_test_episodes,
                                   discard_incomplete_paths=True, render=render)
            model.stochastic_parameters = True

            finish = datetime.datetime.now()
            print("Profiling took: ", finish - start)

            eval_rew = np.mean([np.sum(p['rewards']) for p in ps])
            eval_std = np.std([np.sum(p['rewards']) for p in ps])
            print('Epoch {}, Eval Reward: {}, Std: {}'.format(epoch, eval_rew, eval_std))

            rews = [np.sum(p['rewards']) for p in ps]
            blist = [1 if k > -200 else 0 for k in rews]
            perc = int(dtype_default(sum(blist)) / dtype_default(len(blist)) * 100)

            print(f"Success Rate: {perc:.2f}%")

            # TODO: set
            # if saving_policies:
            #     print('Saved')
            #     torch.save(model.state_dict(),
            #                'offline_rl/{}/_{}_{}_{}_{}_{}p.pt'.format(args.save_dir, name, args.gp_type, epoch,
            #                                                       int(eval_rew), perc))

#%%

    ####### Saving model parameters #######
    if save:
        architecture = ''
        for i in kwargs['architecture']:
            architecture = architecture + str(i) + '_'
        architecture = architecture[:-1]

        with open(f'saved_models/offline_rl/params_pickle', 'wb') as file:
        # with open(f'saved_models/offline_rl/params_pickle_{architecture}', 'wb') as file:
            pickle.dump(params, file)