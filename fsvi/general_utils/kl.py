from jax import numpy as jnp


def jax_kl_general(mean_p, mean_q, cov_p, cov_q, noise=0.0, breakdown=False):
    """
    calculate KL(p || q)
    """
    # convert from float32 to float64 to avoid overflow error in matrix multiplicaiton
    # mean_p, mean_q, cov_p, cov_q = list(map(lambda x: np.array(x).astype(np.float64), [mean_p, mean_q, cov_p, cov_q]))

    noise_matrix = jnp.eye(cov_q.shape[0], dtype=jnp.float32) * noise
    cov_q += noise_matrix
    cov_p += noise_matrix

    D = mean_q.shape[0]
    diff = mean_q - mean_p

    # np.linalg.det uses Cholesky decomposition
    det_term = jax_robust_det_kl_term_numpy(cov_p=cov_p, cov_q=cov_q)
    inv_cov_q = jnp.linalg.inv(cov_q)
    tr_term = jnp.trace(jnp.matmul(inv_cov_q, cov_p))
    quad_term = jnp.matmul(jnp.matmul(diff.T, inv_cov_q), diff)
    # print("inv_cov_q", inv_cov_q)
    # print("singular values of inv_cov_q", jnp.linalg.svd(inv_cov_q)[1])
    # print("singular values of cov_p", jnp.linalg.svd(cov_p)[1])
    # print("tr_term", tr_term)
    # print("det_term", det_term)
    # print("quad_term", quad_term)
    # print("D", D)
    kl = 0.5 * (tr_term + det_term + quad_term - D)
    if breakdown:
        return kl, {"det_term": det_term, "tr_term": tr_term, "quad_term": quad_term}
    else:
        return kl


def jax_kl_diag(mean_q, mean_p, cov_q, cov_p, noise=0.0) -> float:
    """
    calculate KL(q || p)
    """
    noise_vector = jnp.ones(cov_q.shape[0], dtype=jnp.float32) * noise

    D = mean_q.shape[0]
    cov_p, cov_q = map(jnp.diag, [cov_p, cov_q])
    cov_q += noise_vector
    cov_p += noise_vector

    diff = mean_q - mean_p

    det_term = jnp.log(jnp.abs(cov_q)).sum() - jnp.log(jnp.abs(cov_p)).sum()
    inv_cov_q = cov_q ** (-1)
    tr_term = (inv_cov_q * cov_p).sum()
    quad_term = ((diff ** 2) * inv_cov_q).sum()
    kl = 0.5 * (tr_term + det_term + quad_term - D)
    return kl


def jax_robust_det_kl_term_numpy(cov_p, cov_q):
    """
    Calculate log(det(cov_q) / det(cov_p)) in a way that avoids infinite determinant
    """
    sign, log_det = jnp.linalg.slogdet(cov_p)
    det_term_p = sign * log_det
    sign, log_det = jnp.linalg.slogdet(cov_q)
    det_term_q = sign * log_det
    det_term = det_term_q - det_term_p
    return det_term


def jax_kl_multioutput(mean_p, mean_q, cov_p, cov_q, noise=0.0, diag_kl=False, breakdown=False):
    function_KL = 0
    ndim = cov_p.ndim
    terms = []
    for i in range(cov_p.shape[-1]):
        cov_p_tp = cov_p[:, :, i, i] if ndim == 4 else cov_p[:, :, i]
        mean_p_tp = mean_p[:, i]
        cov_q_tp = cov_q[:, :, i, i] if ndim == 4 else cov_q[:, :, i]
        mean_q_tp = mean_q[:, i]
        if diag_kl:
            kl = jax_kl_diag(mean_p_tp, mean_q_tp, cov_p_tp, cov_q_tp, noise=noise)
        else:
            kl = jax_kl_general(mean_p_tp, mean_q_tp, cov_p_tp, cov_q_tp, noise=noise, breakdown=breakdown)
            if breakdown:
                terms.append(kl[1])
                kl = kl[0]
        function_KL += kl
    return function_KL if not breakdown else (function_KL, terms)
