import json
from functools import partial

import numpy as np
from ivmodels.constants import DATA_PATH
from ivmodels.simulate import simulate_guggenberger12
from ivmodels.tests import (
    anderson_rubin_test,
    conditional_likelihood_ratio_test,
    lagrange_multiplier_test,
    likelihood_ratio_test,
    wald_test,
)

from ivmodels_experiments.encode import NumpyEncoder
from ivmodels_simulations.tests import lagrange_multiplier_test_liml

output = DATA_PATH / "guggenberger12_power"
output.mkdir(parents=True, exist_ok=True)

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

n_seeds = 1000
n_betas = 500

betas = np.linspace(0.5, 1.5, n_betas)

p_values = {test_name: np.zeros((n_seeds, n_betas)) for test_name in tests}

for seed in range(n_seeds):
    Z, X, y, _, W, beta = simulate_guggenberger12(
        n, k, seed=seed, return_beta=True, h12=10
    )

    for test_name, test in tests.items():
        for beta_idx, beta_value in enumerate(betas):
            _, p_values[test_name][seed, beta_idx] = test(
                Z=Z, X=X, y=y, beta=np.array([beta_value]), W=W, fit_intercept=False
            )

with open(output / "guggenberger_12_power.json", "w+") as f:
    json.dump(p_values, f, cls=NumpyEncoder)
