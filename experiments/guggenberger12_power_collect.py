import os
from functools import partial

# isort: off
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# isort: on

import click
import numpy as np
from ivmodels.simulate import simulate_guggenberger12
from ivmodels.tests import (
    anderson_rubin_test,
    conditional_likelihood_ratio_test,
    lagrange_multiplier_test,
    likelihood_ratio_test,
    wald_test,
)

from ivmodels_simulations.tests import lagrange_multiplier_test_liml

tests = {
    "AR": anderson_rubin_test,
    "AR (GKM)": partial(anderson_rubin_test, critical_values="guggenberger2019more"),
    "CLR": conditional_likelihood_ratio_test,
    "LM": lagrange_multiplier_test,
    "LM (LIML)": lagrange_multiplier_test_liml,
    "LR": likelihood_ratio_test,
    "Wald (LIML)": partial(wald_test, estimator="liml"),
    "Wald (TSLS)": wald_test,
}

mx = 1
mw = 1

beta = np.array([[1]])
gamma = np.array([[1]])

# With k=10, n=1000, n_seeds=10, n_betas=500, one core, this takes 35s on my macbook.
n_seeds = 10000
n_betas = 500

betas = np.linspace(0.5, 1.5, n_betas)

p_values = {test_name: np.zeros((n_seeds, n_betas)) for test_name in tests}


def _run(seed, n, k):
    p_values = {test_name: np.zeros(n_betas) for test_name in tests}
    Z, X, y, _, W = simulate_guggenberger12(n, k=k, seed=seed, h12=10)

    for test_name, test in tests.items():
        for beta_idx, beta_value in enumerate(betas):
            _, p_values[test_name][beta_idx] = test(
                Z=Z, X=X, y=y, beta=np.array([beta_value]), W=W, fit_intercept=False
            )
    return p_values


@click.command()
@click.option("--n", default=1000)
@click.option("--k", default=10)
@click.option("--n_cores", default=1)
def main(n, k, n_cores):
    import json
    import multiprocessing

    from ivmodels_simulations.constants import DATA_PATH
    from ivmodels_simulations.encode import NumpyEncoder

    output = DATA_PATH / "guggenberger12_power"
    output.mkdir(parents=True, exist_ok=True)

    if n_cores == -1:
        n_cores = multiprocessing.cpu_count() - 1

    if n_cores > 1:
        pool = multiprocessing.Pool(n_cores)
        result = pool.map(partial(_run, n=n, k=k), range(n_seeds))
    else:
        result = [_run(seed, n, k) for seed in range(n_seeds)]

    p_values = {test_name: np.zeros((n_seeds, n_betas)) for test_name in tests}

    for seed in range(n_seeds):
        for test_name in tests:
            p_values[test_name][seed, :] = result[seed][test_name]

    with open(output / f"guggenberger_12_power_n={n}_k={k}.json", "w+") as f:
        json.dump(p_values, f, cls=NumpyEncoder)


if __name__ == "__main__":
    main()
