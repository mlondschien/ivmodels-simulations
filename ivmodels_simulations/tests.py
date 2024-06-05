import numpy as np
import scipy
from ivmodels import KClass
from ivmodels.tests.utils import _check_test_inputs
from ivmodels.utils import proj


def lagrange_multiplier_test_liml(*, Z, X, y, beta, W, fit_intercept):
    "Incorrect subset LM test via plugging in the LIML."

    Z, X, y, W, _, beta = _check_test_inputs(Z, X, y, W=W, beta=beta)

    n = X.shape[0]
    liml = KClass(kappa="liml", fit_intercept=fit_intercept).fit(
        X=W, y=y - X @ beta, Z=Z
    )

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
        (n - Z.shape[1] - fit_intercept)
        * residuals_proj_St.T
        @ residuals_proj_St
        / sigma_hat
    )
    p_value = 1 - scipy.stats.chi2(df=X.shape[1]).cdf(statistic)
    return statistic, p_value
