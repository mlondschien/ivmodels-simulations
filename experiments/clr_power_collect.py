import itertools
import multiprocessing
import os
from functools import partial

from ivmodels.utils import oproj
import scipy

# isort: off
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# isort: on

import click
import h5py
import numpy as np
from ivmodels.tests import conditional_likelihood_ratio_test

from ivmodels_simulations.constants import DATA_PATH

output = DATA_PATH / "clr"
output.mkdir(parents=True, exist_ok=True)

tests = {
    "CLR (new)": conditional_likelihood_ratio_test,
    "CLR (old)": partial(
        conditional_likelihood_ratio_test, critical_values="moreira2003conditional"
    ),
}

data_type = np.uint16


def _run(lambda_1, lambda_2, beta, n, k, m, n_seeds):
    p_values = {test_name: np.zeros(n_seeds, dtype=data_type) for test_name in tests}

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)

        Pi_X = np.empty((k, 0))

        cov = np.eye(m+1)
        cov[0, 1] = -0.5
        cov[1, 0] = -0.5

        cond_covariance = cov[1:, 1:] - cov[1:, 0:1] @ cov[0:1, 1:]
        Lambda = np.diag([lambda_1] + [lambda_2] * (m - 1))
        mat = scipy.linalg.sqrtm(cond_covariance)
        mat = mat @ Lambda @ mat

        for idx in range(m):
            Pi = rng.normal(0, 1, k)
            Pi -= Pi.mean(axis=0)
            Pi = oproj(Pi_X, Pi)
            Pi = Pi / np.linalg.norm(Pi)
            # Pi = np.sqrt(mat[idx, idx]) * Pi
            # for j in range(idx):
            #     Pi += mat[idx, j] / np.sqrt(mat[j, j]) * Pi_X[:, j]
            Pi_X = np.hstack([Pi_X, Pi.reshape(-1, 1)])

        Pi_X = Pi_X @ np.linalg.cholesky(mat).T / np.sqrt(n)

        noise = scipy.stats.multivariate_normal.rvs(
            cov=cov,
            size=n,
            random_state=rng,
        )

        Z = rng.normal(0, 1, (n, k))
        X = Z @ Pi_X + noise[:, 1:]
        y = noise[:, 0]

        y_orth = oproj(Z, y)
        X_tilde = X - y.reshape(-1, 1) @ (y_orth.reshape(-1, 1).T @ X) / (y_orth.T @ y_orth)


        for test_name, test in tests.items():
            _, p_value = test(Z=Z, X=X, y=y, beta=beta, fit_intercept=False)
            p_value = p_value * np.iinfo(data_type).max
            p_values[test_name][seed] = p_value

    return p_values


@click.command()
@click.option("--n", default=1000)
@click.option("--k", default=5)
@click.option("--m", default=2)
@click.option("--n_vars", default=21)
@click.option("--n_cores", default=-1)
@click.option("--lambda_max", default=100)
@click.option("--n_seeds", default=5_000)
@click.option("--lambda_1", default=10)
def main(n, k, m, n_vars, n_cores, lambda_max, n_seeds, lambda_1):

    lambda_1s = [lambda_1]
    lambda_2s = np.geomspace(1, lambda_max, n_vars)
    # lambda_2s = np.geomspace(lambda_max, lambda_max, n_vars)

    beta = np.zeros(m)
    beta[0] = 1
    betas = [beta * const for const in np.linspace(-1, 1, 41)]

    if n_cores == -1:
        n_cores = multiprocessing.cpu_count() - 1

    pool = multiprocessing.Pool(n_cores)
    run = partial(_run, n=n, k=k, m=m, n_seeds=n_seeds)
    result = [run(*x) for x in itertools.product(lambda_1s, lambda_2s, betas)]
    #result = pool.starmap(run, itertools.product(lambda_1s, lambda_2s, betas))

    p_values = {
        test_name: np.zeros((n_seeds, n_vars, len(betas)), dtype=data_type)
        for test_name in tests
    }

    for idx, (
        # (lambda_1_idx, _),
        (lambda_2_idx, _),
        (beta_idx, _),
    ) in enumerate(itertools.product(enumerate(lambda_2s), enumerate(betas))):
        for test_name in tests:
            p_values[test_name][:, lambda_2_idx, beta_idx] = result[idx][test_name]

    f = h5py.File(
        output
        / f"clr_power_n={n}_k={k}_m={m}_n_seeds={n_seeds}_n_vars={n_vars}_lambda_max={lambda_max}_lambda_1{lambda_1}.h5",
        "w",
    )
    for test_name in tests:
        grp = f.create_group(test_name)
        grp.create_dataset("p_values", data=p_values[test_name])


# With n=1000, k=100, n_seeds=1, n_taus=n_lambdas=20, this takes 8min on my macbook.
if __name__ == "__main__":
    main()
