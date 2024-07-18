import numpy as np
import scipy
from ivmodels import KClass
from ivmodels.tests.utils import _check_test_inputs
from ivmodels.utils import proj

from ivmodels.tests.lagrange_multiplier import _LM

from scipy.optimize.optimize import MemoizeJac
import matplotlib.pyplot as plt


def lagrange_multiplier_test_liml(*, Z, X, y, beta, W, fit_intercept):
    "Incorrect subset LM test via plugging in the LIML."

    Z, X, y, W, _, beta = _check_test_inputs(Z, X, y, W=W, beta=beta)
    n = X.shape[0]

    lm = _LM(
        X=X,
        y=y,
        W=W,
        Z=Z,
        dof=n - Z.shape[1]
    )

    liml = KClass(kappa="liml", fit_intercept=fit_intercept).fit(
        X=W, y=y - X @ beta, Z=Z
    )
    gamma = np.array([1])
    gamma = liml.coef_

    for _ in range(1):
        statistic, dlm, ddlm = lm.derivative(beta, gamma)
        gamma = gamma - dlm / ddlm

    statistic, _, _ = lm.lm(beta, gamma)
    # residuals = y - X @ beta - W @ liml.coef_
    # residuals_proj = proj(Z, residuals)
    # residuals_orth = residuals - residuals_proj
    # sigma_hat = residuals_orth.T @ residuals_orth

    # S = np.hstack((X, W))
    # S_proj = proj(Z, S)
    # Sigma = residuals_orth.T @ S / sigma_hat
    # St_proj = S_proj - np.outer(residuals_proj, Sigma)

    # residuals_proj_St = proj(St_proj, residuals)

    # statistic = (
    #     (n - Z.shape[1] - fit_intercept)
    #     * residuals_proj_St.T
    #     @ residuals_proj_St
    #     / sigma_hat
    # )
    p_value = 1 - scipy.stats.chi2(df=X.shape[1]).cdf(statistic)
    return statistic, p_value




def lagrange_multiplier_test_one_step(*, Z, X, y, beta, W, C, fit_intercept, gamma0="liml", ddlm="truth", n_steps=1):
    "Incorrect subset LM test via one-step optimization from LIML."

    Z, X, y, W, C, beta = _check_test_inputs(Z, X, y, W=W, beta=beta, C=C)

    if C.shape[1] > 0:
        raise ValueError

    n = X.shape[0]

    lm = _LM(
        X=X,
        y=y,
        W=W,
        Z=Z,
        dof=n - Z.shape[1]
    )


    liml = KClass(kappa="liml", fit_intercept=fit_intercept).fit(
        X=W, y=y - X @ beta, Z=Z
    )

    if isinstance(gamma0, str) and gamma0 == "liml":
        gamma = liml.coef_
    else: 
        gamma = gamma0

    for _ in range(n_steps):

        residuals = y - X @ beta - W @ gamma
        residuals_proj = proj(Z, residuals)
        residuals_orth = residuals - residuals_proj
        sigma_hat = residuals_orth.T @ residuals_orth

        W_proj = proj(Z, W)
        W_orth = W - W_proj
    
        Sigma = residuals_orth.T @ W / sigma_hat
        Wt_proj = W_proj - np.outer(residuals_proj, Sigma)
        Wt_orth = W_orth - np.outer(residuals_orth, Sigma)
    
        residuals_proj_Wt = proj(Wt_proj, residuals)
        kappa = (residuals_proj - residuals_proj_Wt).T @ residuals_proj_Wt / sigma_hat

        statistic, dlm, _ddlm = lm.derivative(beta, gamma)
        if ddlm == "ar":
            _ddlm = 2 * Wt_proj.T @ Wt_proj / sigma_hat * (n - Z.shape[1])
        elif ddlm == "lm":
            _ddlm = 2 * (Wt_proj.T @ Wt_proj - kappa * Wt_orth.T @ Wt_orth) / sigma_hat * (n - Z.shape[1])
        
        gamma = gamma - dlm / _ddlm


    residuals = y - X @ beta - W @ gamma.flatten()
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
    return statistic, p_value, gamma.flatten()
