from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from ivmodels import KClass
from ivmodels.confidence_set import ConfidenceSet
from ivmodels.quadric import Quadric
from ivmodels.tests import (
    anderson_rubin_test,
    conditional_likelihood_ratio_test,
    inverse_anderson_rubin_test,
    inverse_conditional_likelihood_ratio_test,
    inverse_lagrange_multiplier_test,
    inverse_wald_test,
    j_test,
    lagrange_multiplier_test,
    rank_test,
    wald_test,
)

data_dir = Path(__file__).parents[2] / "applications_data" / "tanaka2006risk"
data_dir.mkdir(exist_ok=True, parents=True)

risk = pd.read_stata(data_dir / "20060431_risk.dta")
risk["rainfallxheadnowork"] = risk["rainfall"] * risk["headnowork"]
risk["lograinfall"] = risk["rainfall"].apply(np.log)

outcome = "vfctnc"
exogenous_names = ["chinese", "age", "gender", "edu", "market", "south"]
endogenous_names = ["nmlrlincome", "mnincome"]
instrument_names = ["rainfall", "headnowork"]

exogenous_instrument_list = [
    (exogenous_names, instrument_names),
    (exogenous_names, instrument_names + ["rainfallxheadnowork"]),
]

tests = [
    ("Wald (TSLS)", wald_test, inverse_wald_test),
    (
        "Wald (LIML)",
        partial(wald_test, estimator="liml"),
        partial(inverse_wald_test, estimator="liml"),
    ),
    ("AR", anderson_rubin_test, inverse_anderson_rubin_test),
    ("LM (ours)", partial(lagrange_multiplier_test), inverse_lagrange_multiplier_test),
    (
        "CLR",
        conditional_likelihood_ratio_test,
        inverse_conditional_likelihood_ratio_test,
    ),
]
index = [("estimate (TSLS)", ""), ("estimate (LIML)", "")]

for test_name, _, _ in tests:
    index += [(test_name, "stat. | $p$-value"), (test_name, "conf. set")]
index += [("rank", "stat. | $p$-value"), ("$J_\\liml$", "stat. | $p$-value")]

index = pd.MultiIndex.from_tuples(index, names=["first", "second"])
columns = pd.MultiIndex.from_tuples(
    [(idx, jdx) for idx in range(len(exogenous_instrument_list)) for jdx in range(2)]
)
table = pd.DataFrame(columns=columns, index=index)

df = risk
x_var = ["mnincome"]
w_var = ["nmlrlincome"]

for idx, (exogenous, instrument) in enumerate(exogenous_instrument_list):
    X = df[x_var]
    W = df[w_var]
    XW = df[x_var + w_var]
    y = df[outcome]
    C = df[exogenous]
    Z = df[instrument]

    for est in "TSLS", "LIML":
        model = KClass(kappa=est, fit_intercept=True)
        model.fit(X=XW, y=y, Z=Z, C=C)

        table.loc[(f"estimate ({est})", ""), (idx, 0)] = f"{model.coef_[0]:.3g}"
        table.loc[
            (f"estimate ({est})", ""), (idx, 1)
        ] = f"({np.abs(model.coef_[0]) / np.sqrt(wald_test(X=X, W=W, y=y, Z=Z, C=C, beta=np.zeros(1), estimator=est)[0]):.3g})"

    for test_name, test, inverse_test in tests:
        stat, p_value = test(X=X, W=W, y=y, Z=Z, C=C, beta=np.zeros(1))
        table.loc[(test_name, "stat. | $p$-value"), (idx, 0)] = f"{stat:0.3g}"
        table.loc[(test_name, "stat. | $p$-value"), (idx, 1)] = f"{p_value:.3g}"

        conf_set = inverse_test(X=X, W=W, y=y, Z=Z, C=C, alpha=0.05)
        if isinstance(conf_set, Quadric):
            conf_set = ConfidenceSet.from_quadric(conf_set)
        table.loc[(test_name, "conf. set"), (idx, 0)] = f"{conf_set:.3g}"

    stat, p_value = rank_test(X=XW, Z=Z, C=C)
    table.loc[("rank", "stat. | $p$-value"), (idx, 0)] = f"{stat:0.2f}"
    table.loc[("rank", "stat. | $p$-value"), (idx, 1)] = f"{p_value:.3g}"
    stat, p_value = j_test(X=XW, y=y, Z=Z, C=C, estimator="liml")
    table.loc[("$J_\\liml$", "stat. | $p$-value"), (idx, 0)] = f"{stat:0.2f}"
    table.loc[("$J_\\liml$", "stat. | $p$-value"), (idx, 1)] = f"{p_value:.3g}"

print(table.to_markdown())
print(table.to_latex())
