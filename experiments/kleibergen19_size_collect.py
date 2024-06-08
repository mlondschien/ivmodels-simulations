import itertools
import json
import multiprocessing
import os
from functools import partial

# isort: off
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# isort: on

import numpy as np
import scipy
from ivmodels.simulate import simulate_guggenberger12
from ivmodels.tests import (
    anderson_rubin_test,
    conditional_likelihood_ratio_test,
    lagrange_multiplier_test,
    likelihood_ratio_test,
    wald_test,
)

from ivmodels_simulations.constants import DATA_PATH
from ivmodels_simulations.encode import NumpyEncoder
from ivmodels_simulations.tests import lagrange_multiplier_test_liml

output = DATA_PATH / "kleibergen19_size"
output.mkdir(parents=True, exist_ok=True)

tests = {
    "LM (LIML)": lagrange_multiplier_test_liml,
    "Wald (TSLS)": wald_test,
    "Wald (LIML)": partial(wald_test, estimator="liml"),
    "LR": likelihood_ratio_test,
    "LM": lagrange_multiplier_test,
    "AR": anderson_rubin_test,
    "CLR": conditional_likelihood_ratio_test,
    "AR (Guggenberger)": partial(
        anderson_rubin_test, critical_values="guggenberger2019more"
    ),
    # "CLR (us)": partial(conditional_likelihood_ratio_test, critical_values="us"),
}

# With n=1000, k=100, n_seeds=1, n_taus=n_lambdas=20, this takes 8min on my macbook.
n = 1000
n_seeds = 1

n_taus = 20
n_lambda_1s = 20
n_lambda_2s = 20

mw = 1
mx = 1
m = mx + mw
k = 100

beta = np.array([[1]])
gamma = np.zeros((mw, 1))
beta_gamma = np.concatenate([beta, gamma], axis=0)
cov = np.diag(np.ones(m + 1))
# cov = np.array([[1, 0, 0.95], [0, 1, 0.3], [0.95, 0.3, 1]])
sqrt_cond_cov = scipy.linalg.sqrtm(np.linalg.inv(cov)[1:, 1:])


def _run(tau, lambda_1, lambda_2):
    p_values = {test_name: np.zeros(n_seeds) for test_name in tests}

    Lambda = np.array([[np.sqrt(lambda_1), 0], [0, np.sqrt(lambda_2)]])
    R = np.array([[np.cos(tau), -np.sin(tau)], [np.sin(tau), np.cos(tau)]])
    concentration = sqrt_cond_cov.T @ R @ Lambda.T @ Lambda @ R.T @ sqrt_cond_cov

    for seed in range(n_seeds):
        h11, h12 = concentration[0, 0], concentration[1, 1]
        if np.isclose(h11 * h12, 0):
            rho = 0
        else:
            rho = concentration[0, 1] / np.sqrt(h11 * h12)

        Z, X, y, _, W, beta = simulate_guggenberger12(
            n, k=k, seed=seed, return_beta=True, h11=h11, h12=h12, rho=rho, cov=cov
        )

        for test_name, test in tests.items():
            _, p_value = test(Z=Z, X=X, y=y, W=W, beta=beta, fit_intercept=False)
            p_values[test_name][seed] = p_value

    return p_values


if __name__ == "__main__":

    lambda_max = 20
    taus = np.linspace(0, 2 * np.pi * (n_taus - 1) / n_taus, n_taus)
    lambda_1s = np.linspace(0, lambda_max, n_lambda_1s)
    lambda_2s = np.linspace(0, lambda_max, n_lambda_2s)

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    result = [_run(*x) for x in itertools.product(taus, lambda_1s, lambda_2s)]
    result = pool.starmap(_run, itertools.product(taus, lambda_1s, lambda_2s))

    p_values = {
        test_name: np.zeros((n_seeds, n_taus, n_lambda_1s, n_lambda_2s))
        for test_name in tests
    }

    for idx, (
        (tau_idx, tau),
        (lambda_1_idx, lambda_1),
        (lambda_2_idx, lambda_2),
    ) in enumerate(
        itertools.product(enumerate(taus), enumerate(lambda_1s), enumerate(lambda_2s))
    ):
        for test_name in tests:
            p_values[test_name][:, tau_idx, lambda_1_idx, lambda_2_idx] = result[idx][
                test_name
            ]

    with open("kleibergen19_size.json", "w") as f:
        json.dump(p_values, f, cls=NumpyEncoder)
