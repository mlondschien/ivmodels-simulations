import multiprocessing
import os
from functools import partial

# isort: off
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# isort: on

import click
import numpy as np
import pandas as pd
from ivmodels.simulate import simulate_guggenberger12
from ivmodels.tests import (
    anderson_rubin_test,
    conditional_likelihood_ratio_test,
    lagrange_multiplier_test,
    likelihood_ratio_test,
    wald_test,
)

from ivmodels_simulations.tests import lagrange_multiplier_test_liml

wald_test_liml = partial(wald_test, estimator="liml")
tests = {
    "AR": anderson_rubin_test,
    "AR (GKM)": partial(anderson_rubin_test, critical_values="guggenberger2019more"),
    "CLR": conditional_likelihood_ratio_test,
    "LM": lagrange_multiplier_test,
    "LM (LIML)": lagrange_multiplier_test_liml,
    "LR": likelihood_ratio_test,
    "Wald (LIML)": wald_test_liml,
    "Wald (TSLS)": wald_test,
}

mx = 1
mw = 1
m = mx + mw


def _run(seed, n, k):
    Z, X, y, _, W, beta = simulate_guggenberger12(n=n, k=k, seed=seed, return_beta=True)
    return {
        test_name: test(Z=Z, X=X, y=y, W=W, beta=beta, fit_intercept=False)[1]
        for test_name, test in tests.items()
    }


@click.command()
@click.option("--n", default=1000)
@click.option("--n_cores", default=-1)
def main(n, n_cores):
    import itertools

    from ivmodels_simulations.constants import DATA_PATH

    output = DATA_PATH / "guggenberger12_size"
    output.mkdir(parents=True, exist_ok=True)

    # With n_seeds=100, on one core, this takes ~7s on my macbook.
    n_seeds = 10000
    ks = [5, 10, 15, 20, 25, 30]

    if n_cores == -1:
        n_cores = multiprocessing.cpu_count() - 1

    pool = multiprocessing.Pool(n_cores)
    result = pool.starmap(partial(_run, n=n), itertools.product(range(n_seeds), ks))

    p_values = {(test_name, k): np.zeros(n_seeds) for test_name in tests for k in ks}
    for idx, (seed, k) in enumerate(itertools.product(range(n_seeds), ks)):
        for test_name in tests:
            p_values[(test_name, k)][seed] = result[idx][test_name]

    columns = pd.MultiIndex.from_tuples(p_values.keys())
    pd.DataFrame(p_values, columns=columns).to_csv(
        output / f"guggenberger12_p_values_n={n}.csv", index=False
    )


if __name__ == "__main__":
    main()
