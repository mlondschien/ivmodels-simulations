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

from ivmodels_simulations.constants import DATA_PATH
from ivmodels_simulations.tests import lagrange_multiplier_test_liml

output = DATA_PATH / "guggenberger12_size"
output.mkdir(parents=True, exist_ok=True)

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
n_seeds = 10000
ks = [5, 10, 15, 20, 25, 30]

p_values = {(test_name, k): np.zeros(n_seeds) for test_name in tests for k in ks}

for k_idx, k in enumerate(ks):
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)

        Z, X, y, _, W, beta = simulate_guggenberger12(n, k, seed=seed, return_beta=True)

        for test_name, test in tests.items():
            _, p_value = test(Z, X, y, W=W, beta=np.ones((mx, 1)), fit_intercept=False)
            p_values[(test_name, k)][seed] = p_value

pd.DataFrame(p_values).to_csv(output / "guggenberger12_p_values.csv", index=False)
