import numpy as np
import sklearn
import sklearn.metrics
import scipy
from scipy.linalg import cholesky, cho_solve, solve

import jax
import jax.numpy as jnp
from jax.config import config

from tensorflow_probability.substrates import jax as tfp
tfpk = tfp.math.psd_kernels

from fsvi.utils.utils import sigma_transform, zero_mean_transform, delta_vjp_jvp

# config.update("jax_enable_x64", False)


def GP_Classification_Laplace(
    gp_train_x, gp_train_y, inducing_inputs, output_dim
):  # TODO: Make sure output dim is provided when calling function
    gp_train_y = jnp.asarray(gp_train_y.T)
    gp_train_x = jnp.asarray(gp_train_x)
    inducing_inputs = jnp.asarray(inducing_inputs)
    gp_n_train = gp_train_x.shape[0]
    mes_size = inducing_inputs.shape[0]

    gamma = 0.01
    K = sklearn.metrics.pairwise.rbf_kernel(gp_train_x, gp_train_x, gamma)
    K_star = sklearn.metrics.pairwise.rbf_kernel(gp_train_x, inducing_inputs, gamma)
    K_star_star = sklearn.metrics.pairwise.rbf_kernel(inducing_inputs, inducing_inputs, gamma)

    K_all = np.zeros((gp_n_train * output_dim, gp_n_train * output_dim))
    for i in range(10):
        K_all[
            i * gp_n_train : (i + 1) * gp_n_train,
            i * gp_n_train : (i + 1) * gp_n_train,
        ] = K

    f = np.zeros((gp_n_train, output_dim)) + 0.1
    R = np.eye(gp_n_train)
    for _ in range(9):
        R = np.vstack((R, np.eye(gp_n_train)))

    for _ in range(10):
        f = f.reshape([output_dim, gp_n_train])
        pi = scipy.special.softmax(f, axis=0).flatten()

        E = np.zeros((gp_n_train * output_dim, gp_n_train * output_dim))
        Big_pi = np.zeros((gp_n_train, gp_n_train))
        for i in range(10):
            D = np.diag(pi[i * gp_n_train : (i + 1) * gp_n_train])
            Big_pi = np.vstack((Big_pi, D))
            D_sr = np.sqrt(D)
            B = np.eye(D.shape[0]) + D_sr @ K @ D_sr
            L = cholesky(B, lower=True)
            E[
                i * gp_n_train : (i + 1) * gp_n_train,
                i * gp_n_train : (i + 1) * gp_n_train,
            ] = D_sr @ cho_solve((L, True), D_sr)

        Big_pi = Big_pi[gp_n_train:, :]
        B2 = np.zeros((gp_n_train, gp_n_train))
        for i in range(10):
            B2 = (
                B2
                + E[
                    i * gp_n_train : (i + 1) * gp_n_train,
                    i * gp_n_train : (i + 1) * gp_n_train,
                ]
            )
        M = cholesky(B2)
        D = np.diag(pi)
        f_ = f.flatten()
        b = (D - Big_pi @ Big_pi.T) @ f_ + gp_train_y.flatten() - pi
        c = E @ K_all @ b

        a = b - c + E @ R @ cho_solve((M, True), R.T @ c)
        f = K_all @ a

    pi = pi.reshape([output_dim, gp_n_train])
    pred_label = pi.argmax(0)
    true_label = gp_train_y.argmax(0)

    mu_star = np.zeros((mes_size, output_dim))
    Sigma_star = np.zeros((mes_size, output_dim))
    for i in range(10):
        yc = gp_train_y[i, :]
        pi_c = pi[i, :]
        mu_star[:, i] = np.squeeze((yc - pi_c).reshape([1, -1]) @ K_star)

        E_c = E[
            i * gp_n_train : (i + 1) * gp_n_train,
            i * gp_n_train : (i + 1) * gp_n_train,
        ]
        b = E_c @ K_star
        c = E_c @ cho_solve((M, True), b)
        Sigma_star[:, i] = Sigma_star[:, i] + np.diag(
            K_star_star - b.T @ K_star + c.T @ K_star
        )

    return mu_star, Sigma_star


def GP_Regression_RBF(
    predict_fn,
    params_mean_copy,
    params_log_var_copy,
    rng_key,
    x_condition,
    y_condition,
    inputs,
    noise_std,
    jitter=1e-2,
    cov_diag=True,
    posterior=True
):
    x_condition = jnp.asarray(x_condition)
    amplitude = 10.0
    length_scale = 0.5
    rbf = tfpk.ExponentiatedQuadratic(amplitude=amplitude, length_scale=length_scale)
    K = rbf.matrix(x_condition, x_condition)
    K_star = rbf.matrix(inputs, x_condition)
    K_star_star = rbf.matrix(inputs, inputs)

    K = jnp.squeeze(K)
    K_star = jnp.squeeze(K_star)
    K_star_star = jnp.squeeze(K_star_star)
    N = K.shape[0]
    K_inv = jnp.linalg.inv(K + noise_std ** 2 * jnp.eye(N))

    mean = K_star @ K_inv @ y_condition

    if cov_diag:
        if posterior:
            cov = (
                jnp.diag(K_star_star - jnp.einsum("ij,ij->i", K_star @ K_inv, K_star))
                + jitter
            )
        else:
            cov = jnp.diag(K_star_star) + jitter
    else:
        if posterior:
            cov = (
                K_star_star
                - jnp.einsum("ij,ij->i", K_star @ K_inv, K_star)
                + jitter  # TODO: test
            )
        else:
            cov = K_star_star

    return mean, cov


def GP_Regression_DNN2GP(
    predict_fn,
    params_mean_copy,
    params_log_var_copy,
    rng_key,
    x_condition,
    y_condition,
    inputs,
    noise_std,
    jitter=1e-2,
    cov_diag=True,
    posterior=False,  # TODO: Remove hardcoding
    prior_var=100.  # TODO: Remove hardcoding
):
    def f_predict_test_gp(v):
        return predict_fn(v, params_log_var_copy, inputs, rng=rng_key, stochastic_parameters=False)

    def f_predict_train_gp(v):
        return predict_fn(v, params_log_var_copy, x_condition, rng=rng_key, stochastic_parameters=False)

    delta_dummy_train = jnp.ones_like(x_condition)
    delta_dummy_test = jnp.ones_like(inputs)

    if posterior:
        params_mean = params_mean_copy
        params_var = sigma_transform(params_log_var_copy, isone=True)

        mean = jnp.squeeze(predict_fn(params_mean_copy, params_log_var_copy, inputs, rng=rng_key, stochastic_parameters=False))

        K = jax.jacobian(delta_vjp_jvp, argnums=0)(
            delta_dummy_train,
            f_predict_train_gp,
            f_predict_train_gp,
            params_var,
            params_mean
        )
        K_star = jax.jacobian(delta_vjp_jvp, argnums=0)(
            delta_dummy_train,
            f_predict_test_gp,
            f_predict_train_gp,
            params_var,
            params_mean
        )
        K_star_star = jax.jacobian(delta_vjp_jvp, argnums=0)(
            delta_dummy_test,
            f_predict_test_gp,
            f_predict_test_gp,
            params_var,
            params_mean
        )
        K = jnp.squeeze(K)
        K_star = jnp.squeeze(K_star)
        K_star_star = jnp.squeeze(K_star_star)
        N = K.shape[0]
        K_inv = jnp.linalg.inv(K + noise_std ** 2 * jnp.eye(N))

        if cov_diag:
            cov = (
                jnp.diag(K_star_star - jnp.einsum("ij,ij->i", K_star @ K_inv, K_star))
                + jitter
            )
        else:
            cov = (
                K_star_star
                - jnp.einsum("ij,ij->i", K_star @ K_inv, K_star)
                + jitter  # TODO: test
            )
    else:
        params_mean = zero_mean_transform(params_mean_copy)
        params_var = sigma_transform(params_log_var_copy, isone=True, prior_var=prior_var)

        mean = jnp.squeeze(predict_fn(params_mean, params_var, inputs, rng=rng_key, stochastic_parameters=False))

        K_star_star = jax.jacobian(delta_vjp_jvp, argnums=0)(delta_dummy_test, f_predict_test_gp, f_predict_test_gp, params_var, params_mean)
        K_star_star = jnp.squeeze(K_star_star)
        if cov_diag:
            cov = jnp.diag(K_star_star)
        else:
            cov = K_star_star

    return mean, cov