from functools import partial

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
k = 20
n = 1000

beta = np.array([[1]])
gamma = np.array([[1]])

n_seeds = 10000
n_betas = 500

betas = np.linspace(0.5, 1.5, n_betas)

p_values = {test_name: np.zeros((n_seeds, n_betas)) for test_name in tests}


def _run(seed):
    p_values = {test_name: np.zeros(n_betas) for test_name in tests}
    Z, X, y, _, W = simulate_guggenberger12(n, k=k, seed=seed, h12=10)

    for test_name, test in tests.items():
        for beta_idx, beta_value in enumerate(betas):
            _, p_values[test_name][beta_idx] = test(
                Z=Z, X=X, y=y, beta=np.array([beta_value]), W=W, fit_intercept=False
            )
    return p_values


if __name__ == "__main__":
    import json
    import multiprocessing

    from ivmodels_simulations.constants import DATA_PATH
    from ivmodels_simulations.encode import NumpyEncoder

    output = DATA_PATH / "guggenberger12_power"
    output.mkdir(parents=True, exist_ok=True)

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    result = pool.map(_run, range(n_seeds))

    p_values = {test_name: np.zeros((n_seeds, n_betas)) for test_name in tests}

    for seed in range(n_seeds):
        for test_name in tests:
            p_values[test_name][seed, :] = result[seed][test_name]

    with open(output / "guggenberger_12_power.json", "w+") as f:
        json.dump(p_values, f, cls=NumpyEncoder)
