import itertools
import multiprocessing
import os
from functools import partial

from ivmodels.utils import oproj

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

output = DATA_PATH / "kleibergen19_clr_size"
output.mkdir(parents=True, exist_ok=True)

tests = {
    "CLR (new)": conditional_likelihood_ratio_test,
    "CLR (old)": partial(
        conditional_likelihood_ratio_test, method="numerical_integration"
    ),
}

data_type = np.uint16


def _run(lambda_1, lambda_2, n, k, m, n_seeds):
    p_values = {test_name: np.zeros(n_seeds, dtype=data_type) for test_name in tests}

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)

        Pi_X = np.empty((k, 0))

        for lam in [lambda_1] + [lambda_2] * (m - 1):
            Pi = rng.normal(0, 1, (k, 1))
            Pi -= Pi.mean(axis=0)
            Pi = oproj(Pi_X, Pi)
            Pi = Pi / np.linalg.norm(Pi) / np.sqrt(n) * lam
            Pi_X = np.hstack([Pi_X, Pi])

        Z = rng.normal(0, 1, (n, k))
        X = Z @ Pi_X + rng.normal(0, 1, (n, m))
        y = rng.normal(0, 1, (n, 1))

        for test_name, test in tests.items():
            _, p_value = test(Z=Z, X=X, y=y, beta=np.zeros(m), fit_intercept=False)
            p_value = p_value * np.iinfo(data_type).max
            p_values[test_name][seed] = p_value

    return p_values


@click.command()
@click.option("--n", default=1000)
@click.option("--k", default=5)
@click.option("--m", default=2)
@click.option("--n_vars", default=31)
@click.option("--n_cores", default=-1)
@click.option("--lambda_max", default=1000)
@click.option("--n_seeds", default=50_000)
def main(n, k, m, n_vars, n_cores, lambda_max, n_seeds):

    lambda_1s = np.geomspace(1, lambda_max, n_vars)
    lambda_2s = np.geomspace(1, lambda_max, n_vars)

    if n_cores == -1:
        n_cores = multiprocessing.cpu_count() - 1

    pool = multiprocessing.Pool(n_cores)
    run = partial(_run, n=n, k=k, m=m, n_seeds=n_seeds)
    # result = [run(*x) for x in itertools.product(taus, lambda_1s, lambda_2s)]
    result = pool.starmap(run, itertools.product(lambda_1s, lambda_2s))

    p_values = {
        test_name: np.zeros((n_seeds, n_vars, n_vars), dtype=data_type)
        for test_name in tests
    }

    for idx, (
        (lambda_1_idx, _),
        (lambda_2_idx, _),
    ) in enumerate(itertools.product(enumerate(lambda_1s), enumerate(lambda_2s))):
        for test_name in tests:
            p_values[test_name][:, lambda_1_idx, lambda_2_idx] = result[idx][test_name]

    f = h5py.File(
        output
        / f"clr_size_n={n}_k={k}_m={m}_n_seeds={n_seeds}_n_vars={n_vars}_lambda_max={lambda_max}.h5",
        "w",
    )
    for test_name in tests:
        grp = f.create_group(test_name)
        grp.create_dataset("p_values", data=p_values[test_name])


# With n=1000, k=100, n_seeds=1, n_taus=n_lambdas=20, this takes 8min on my macbook.
if __name__ == "__main__":
    main()
