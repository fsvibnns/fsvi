from functools import partial
from typing import Callable

import haiku as hk
import jax
import tree
from jax import numpy as jnp, jit

from fsvi.utils.utils_linearization import get_ntk, delta_vjp_jvp


@partial(jit, static_argnums=(0,))
def delta_vjp(predict_fn, params_mean, params_var, delta):
    vjp_tp = jax.vjp(predict_fn, params_mean)[1](delta)
    return (tree.map_structure(lambda x1, x2: x1 * x2, params_var, vjp_tp[0]),)


def diag_custom_ntk_for_loop(
    apply_fn: Callable, x: jnp.ndarray, params: hk.Params, sigma: hk.Params, diag=True
):
    """

    @param apply_fn:
    @param x:
    @param params:
    @param sigma:
    @param diag:
    @return:
        diag_ntk_sum_array: array of shape (batch_dim, output_dim)
    """

    assert diag
    cov = []
    for i in range(x.shape[0]):
        fwd_fn = partial(apply_fn, x=x[i:i+1])
        cov.append(
            jnp.diag(
                get_ntk(fwd_fn, delta_vjp_jvp, delta_vjp, params, sigma,)[0, :, 0, :]
            )
        )
    cov = jnp.array(cov)
    return cov
