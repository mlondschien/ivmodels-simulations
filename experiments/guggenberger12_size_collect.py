import multiprocessing
import os
from functools import partial

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

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
n = 1000


def _run(seed, k):
    Z, X, y, _, W, beta = simulate_guggenberger12(n, k=k, seed=seed, return_beta=True)
    return {
        test_name: test(Z=Z, X=X, y=y, W=W, beta=beta, fit_intercept=False)[1]
        for test_name, test in tests.items()
    }


if __name__ == "__main__":
    import itertools

    from ivmodels_simulations.constants import DATA_PATH

    output = DATA_PATH / "guggenberger12_size"
    output.mkdir(parents=True, exist_ok=True)

    n_seeds = 10000
    ks = [5, 10, 15, 20, 25, 30]

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    result = pool.starmap(_run, itertools.product(range(n_seeds), ks))

    p_values = {(test_name, k): np.zeros(n_seeds) for test_name in tests for k in ks}
    for idx, (seed, k) in enumerate(itertools.product(range(n_seeds), ks)):
        for test_name in tests:
            p_values[(test_name, k)][seed] = result[idx][test_name]

    columns = pd.MultiIndex.from_tuples(p_values.keys())
    pd.DataFrame(p_values, columns=columns).to_csv(
        output / "guggenberger12_p_values.csv", index=False
    )
