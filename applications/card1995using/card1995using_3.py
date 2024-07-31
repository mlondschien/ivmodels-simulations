from functools import partial

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
    inverse_likelihood_ratio_test,
    inverse_wald_test,
    j_test,
    lagrange_multiplier_test,
    likelihood_ratio_test,
    rank_test,
    wald_test,
)

from ivmodels_simulations.load import load_card1995using
from ivmodels_simulations.tests import lagrange_multiplier_test_liml

df = load_card1995using()

# All models include a black indicator, indicators for southern residence and residence
# in an SMSA in 1976, indicators for region in 1966 and living in an SMSA in 1966, as
# well as experience and experience squared.
indicators = ["black", "smsa66r", "smsa76r", "reg76r"]
# exclude reg669, as sum(reg661, ..., reg669) = 1
indicators += [f"reg66{i}" for i in range(1, 9)]
exp = ["exp76", "exp762"]
age = ["age76", "age762"]

family = ["daded", "momed", "nodaded", "nomomed", "famed", "momdad14", "sinmom14"]
fs = [f"f{i}" for i in range(1, 9)]  # exclude f9 as sum(f1, ..., f9) = 1
family += fs

df["low_parental_education"] = df["famed"].isin([8, 9])
df["nearc4a_x_low_parental_education"] = df["nearc4a"] * df["low_parental_education"]
df["nearc4b_x_low_parental_education"] = df["nearc4b"] * df["low_parental_education"]
df["nearc2_x_low_parental_education"] = df["nearc2"] * df["low_parental_education"]

# Card, 1995, Table 5, b: In column 4 the  instruments are interactions of 8 parental
# education class indicators with an indicator for living near a college in 1966.
nearc4a_x_fs = [f"nearc4a_x_{f}" for f in fs]
df[nearc4a_x_fs] = df[fs].to_numpy() * df[["nearc4a"]].to_numpy()
nearc4b_x_fs = [f"nearc4b_x_{f}" for f in fs]
df[nearc4b_x_fs] = df[fs].to_numpy() * df[["nearc4b"]].to_numpy()
nearc2_x_fs = [f"nearc2_x_{f}" for f in fs]
df[nearc2_x_fs] = df[fs].to_numpy() * df[["nearc2"]].to_numpy()


exogenous_instrument_list = [
    (indicators, ["nearc4"]),
    (indicators + family, ["nearc4"]),
    (indicators, ["nearc4a", "nearc4b", "nearc2"]),
    (indicators + family, ["nearc4a", "nearc4b", "nearc2"]),
    (
        indicators + family,
        [
            "nearc4a",
            "nearc4b",
            "nearc2",
            "nearc4a_x_low_parental_education",
            "nearc4b_x_low_parental_education",
            "nearc2_x_low_parental_education",
        ],
    ),
    (
        indicators + family,
        ["nearc4a", "nearc4b", "nearc2"] + nearc4a_x_fs + nearc4b_x_fs + nearc2_x_fs,
    ),
]

tests = [
    ("Wald (TSLS)", wald_test, inverse_wald_test),
    (
        "Wald (LIML)",
        partial(wald_test, estimator="liml"),
        partial(inverse_wald_test, estimator="liml"),
    ),
    ("AR", anderson_rubin_test, inverse_anderson_rubin_test),
    # ("AR (GKM)", partial(anderson_rubin_test, critical_values="guggenberger12"), inverse_anderson_rubin_test),
    ("LM (ours)", partial(lagrange_multiplier_test), inverse_lagrange_multiplier_test),
    ("LM (LIML)", lagrange_multiplier_test_liml, inverse_lagrange_multiplier_test),
    ("LR", likelihood_ratio_test, inverse_likelihood_ratio_test),
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

xrange = (-0.3, 0.4)


for idx, (exogenous, instrument) in enumerate(exogenous_instrument_list):
    X = df[["ed76"]]
    XW = X
    y = df["lwage76"]
    C = df[exogenous]
    Z = df[instrument]

    for est in "TSLS", "LIML":
        model = KClass(kappa=est, fit_intercept=True)
        model.fit(X=XW, y=y, Z=Z, C=C)

        table.loc[(f"estimate ({est})", ""), (idx, 0)] = f"{model.coef_[0]:.3g}"
        table.loc[
            (f"estimate ({est})", ""), (idx, 1)
        ] = f"({np.abs(model.coef_[0]) / np.sqrt(wald_test(X=X, y=y, Z=Z, C=C, beta=np.zeros(1), estimator=est)[0]):.3g})"

    for test_name, test, inverse_test in tests:
        stat, p_value = test(X=X, W=None, y=y, Z=Z, C=C, beta=np.zeros(1))
        table.loc[(test_name, "stat. | $p$-value"), (idx, 0)] = f"{stat:0.2f}"
        table.loc[(test_name, "stat. | $p$-value"), (idx, 1)] = f"{p_value:.3g}"

        conf_set = inverse_test(X=X, y=y, Z=Z, C=C, alpha=0.05)
        if isinstance(conf_set, Quadric):
            conf_set = ConfidenceSet.from_quadric(conf_set)
        table.loc[(test_name, "conf. set"), (idx, 0)] = f"{conf_set:.2f}"

    stat, p_value = rank_test(X=XW, Z=Z, C=C)
    table.loc[("rank", "stat. | $p$-value"), (idx, 0)] = f"{stat:0.2f}"
    table.loc[("rank", "stat. | $p$-value"), (idx, 1)] = f"{p_value:.3g}"
    stat, p_value = j_test(X=XW, y=y, Z=Z, C=C, estimator="liml")
    table.loc[("$J_\\liml$", "stat. | $p$-value"), (idx, 0)] = f"{stat:0.2f}"
    table.loc[("$J_\\liml$", "stat. | $p$-value"), (idx, 1)] = f"{p_value:.3g}"

print(table.to_markdown())
print(table.to_latex())
# fig.savefig(FIGURES_PATH / "card_2.pdf", bbox_inches="tight")
