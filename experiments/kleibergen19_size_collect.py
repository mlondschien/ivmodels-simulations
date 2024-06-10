import itertools
import multiprocessing
import os
from functools import partial

# isort: off
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# isort: on

import click
import h5py
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
from ivmodels_simulations.tests import lagrange_multiplier_test_liml

output = DATA_PATH / "kleibergen19_size"
output.mkdir(parents=True, exist_ok=True)

tests = {
    "AR": anderson_rubin_test,
    "AR (Guggenberger)": partial(
        anderson_rubin_test, critical_values="guggenberger2019more"
    ),
    "CLR": conditional_likelihood_ratio_test,
    "LM": lagrange_multiplier_test,
    "LM (LIML)": lagrange_multiplier_test_liml,
    "LR": likelihood_ratio_test,
    "Wald (LIML)": partial(wald_test, estimator="liml"),
    "Wald (TSLS)": wald_test,
    "CLR (us)": partial(conditional_likelihood_ratio_test, critical_values="us"),
}

data_type = np.uint16


def _run(tau, lambda_1, lambda_2, n, k, n_seeds, cov):
    p_values = {test_name: np.zeros(n_seeds, dtype=data_type) for test_name in tests}

    # cov = np.eye(3) but with cov[1, 0] = -beta_0, cov[2, 0] = -gamma_0
    # s.t. mat.T @ cov @ mat = Cov(y - X beta_0 - W gamma_0, X, W) ???
    mat = np.array([[1, 0, 0], [-1, 1, 0], [-1, 0, 1]])

    Sigma = mat.T @ cov @ mat
    sqrt_cond_Sigma = scipy.linalg.sqrtm(np.linalg.inv(Sigma)[1:, 1:])
    Lambda = np.array([[np.sqrt(lambda_1), 0], [0, np.sqrt(lambda_2)]])
    R = np.array([[np.cos(tau), -np.sin(tau)], [np.sin(tau), np.cos(tau)]])
    concentration = sqrt_cond_Sigma.T @ R @ Lambda.T @ Lambda @ R.T @ sqrt_cond_Sigma

    for seed in range(n_seeds):
        h11, h12 = np.sqrt(concentration[0, 0]), np.sqrt(concentration[1, 1])
        if np.isclose(h11 * h12, 0):
            rho = 0
        else:
            rho = np.sqrt(concentration[0, 1]) / np.sqrt(h11 * h12)

        Z, X, y, _, W, beta = simulate_guggenberger12(
            n, k=k, seed=seed, return_beta=True, h11=h11, h12=h12, rho=rho, cov=cov
        )

        for test_name, test in tests.items():
            _, p_value = test(Z=Z, X=X, y=y, W=W, beta=beta, fit_intercept=False)
            p_value = p_value * np.iinfo(data_type).max
            p_values[test_name][seed] = p_value

    return p_values


@click.command()
@click.option("--n", default=1000)
@click.option("--k", default=100)
@click.option("--n_vars", default=20)
@click.option("--n_cores", default=-1)
@click.option("--lambda_max", default=20)
@click.option("--n_seeds", default=1000)
@click.option("--cov_type", default="identity")
def main(n, k, n_vars, n_cores, lambda_max, n_seeds, cov_type):
    n_taus = n_vars
    n_lambda_1s = n_vars
    n_lambda_2s = n_vars

    mw = 1
    mx = 1
    m = mx + mw

    if cov_type == "identity":
        cov = np.diag(np.ones(m + 1))
    elif cov_type == "guggenberger12":
        cov = np.array([[1, 0, 0.95], [0, 1, 0.3], [0.95, 0.3, 1]])
    else:
        raise ValueError(f"Invalid cov_type: {cov_type}")

    taus = np.linspace(0, 2 * np.pi * (n_taus - 1) / n_taus, n_taus)
    lambda_1s = np.linspace(0, lambda_max, n_lambda_1s)
    lambda_2s = np.linspace(0, lambda_max, n_lambda_2s)

    if n_cores == -1:
        n_cores = multiprocessing.cpu_count() - 1

    # pool = multiprocessing.Pool(n_cores)
    run = partial(_run, n=n, k=k, n_seeds=n_seeds, cov=cov)
    result = [run(*x) for x in itertools.product(taus, lambda_1s, lambda_2s)]
    # result = pool.starmap(run, itertools.product(taus, lambda_1s, lambda_2s))

    p_values = {
        test_name: np.zeros(
            (n_seeds, n_taus, n_lambda_1s, n_lambda_2s), dtype=data_type
        )
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

    f = h5py.File(
        output
        / f"kleibergen19_size_n={n}_k={k}_n_seeds={n_seeds}_n_vars={n_vars}_lambda_max={lambda_max}_cov_type={cov_type}.h5",
        "w",
    )
    for test_name in tests:
        grp = f.create_group(test_name)
        grp.create_dataset("p_values", data=p_values[test_name])


# With n=1000, k=100, n_seeds=1, n_taus=n_lambdas=20, this takes 8min on my macbook.
if __name__ == "__main__":
    main()
