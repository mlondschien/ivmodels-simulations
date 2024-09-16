import numpy as np
import scipy
from ivmodels import KClass
from ivmodels.utils import _check_inputs, oproj, proj


def lagrange_multiplier_test_liml(*, Z, X, y, beta, W, C=None, fit_intercept=True):
    "Incorrect subset LM test via plugging in the LIML."

    Z, X, y, W, C, _, beta = _check_inputs(Z, X, y, W=W, beta=beta, C=C)

    n, k = Z.shape

    if fit_intercept:
        C = np.hstack([C, np.ones((n, 1))])

    X, W, Z, y = oproj(C, X, W, Z, y)

    liml = KClass(kappa="liml", fit_intercept=False).fit(X=W, y=y - X @ beta, Z=Z)

    residuals = y - X @ beta - W @ liml.coef_
    residuals_proj = proj(Z, residuals)
    residuals_orth = residuals - residuals_proj
    sigma_hat = residuals_orth.T @ residuals_orth

    S = np.hstack((X, W))
    S_proj = proj(Z, S)
    Sigma = residuals_orth.T @ S / sigma_hat
    St_proj = S_proj - np.outer(residuals_proj, Sigma)

    residuals_proj_St = proj(St_proj, residuals)

    statistic = (
        (n - k - C.shape[1]) * residuals_proj_St.T @ residuals_proj_St / sigma_hat
    )
    p_value = 1 - scipy.stats.chi2(df=X.shape[1]).cdf(statistic)
    return statistic, p_value
