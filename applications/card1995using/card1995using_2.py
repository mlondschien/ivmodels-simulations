from functools import partial

import matplotlib.pyplot as plt
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

from ivmodels_simulations.constants import COLOR_CYCLE, FIGURES_PATH, LINESTYLES_MAPPING
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
    (indicators, ["nearc4a", "nearc4b", "nearc2"] + age),
    (indicators + family, ["nearc4a", "nearc4b", "nearc2"] + age),
    (
        indicators + family,
        [
            "nearc4a",
            "nearc4b",
            "nearc2",
            "nearc4a_x_low_parental_education",
            "nearc4b_x_low_parental_education",
            "nearc2_x_low_parental_education",
        ]
        + age,
    ),
    (
        indicators + family,
        ["nearc4a", "nearc4b", "nearc2"]
        + nearc4a_x_fs
        + nearc4b_x_fs
        + nearc2_x_fs
        + age,
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
    (
        "AR (GKM)",
        partial(anderson_rubin_test, critical_values="guggenberger12"),
        inverse_anderson_rubin_test,
    ),
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

fig, axes = plt.subplots(len(exogenous_instrument_list), figsize=(10, 7))
fig.tight_layout(rect=[0.1, 0.01, 0.8, 0.98])

for idx, (exogenous, instrument) in enumerate(exogenous_instrument_list):
    X = df[["ed76"]]
    W = df[["exp76", "exp762"]]
    XW = df[["ed76", "exp76", "exp762"]]
    y = df["lwage76"]
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
        table.loc[(test_name, "stat. | $p$-value"), (idx, 0)] = f"{stat:0.2f}"
        table.loc[(test_name, "stat. | $p$-value"), (idx, 1)] = f"{p_value:.3g}"

        conf_set = inverse_test(X=X, W=W, y=y, Z=Z, C=C, alpha=0.05)
        if isinstance(conf_set, Quadric):
            conf_set = ConfidenceSet.from_quadric(conf_set)
        table.loc[(test_name, "conf. set"), (idx, 0)] = f"{conf_set:.2f}"

        grid = np.linspace(*xrange, 400)
        values = np.zeros_like(grid)
        for i, beta in enumerate(grid):
            values[i] = test(X=X, W=W, y=y, Z=Z, C=C, beta=np.array([beta]))[1]
        axes[idx].plot(
            grid,
            values,
            label=test_name if idx == 0 else None,
            color=COLOR_CYCLE[test_name],
            linestyle=LINESTYLES_MAPPING[test_name],
        )

    stat, p_value = rank_test(X=XW, Z=Z, C=C)
    table.loc[("rank", "stat. | $p$-value"), (idx, 0)] = f"{stat:0.2f}"
    table.loc[("rank", "stat. | $p$-value"), (idx, 1)] = f"{p_value:.3g}"
    stat, p_value = j_test(X=XW, y=y, Z=Z, C=C, estimator="liml")
    table.loc[("$J_\\liml$", "stat. | $p$-value"), (idx, 0)] = f"{stat:0.2f}"
    table.loc[("$J_\\liml$", "stat. | $p$-value"), (idx, 1)] = f"{p_value:.3g}"

    specifications = {0: "(i)", 1: "(ii)", 2: "(iii)", 3: "(iv)"}
    family = "with" if "daded" in exogenous else "without"
    axes[idx].set_title(
        f"specification {specifications[idx]}, k={Z.shape[1]}, {family} family background variables",
        fontsize=12,
    )
    axes[idx].set_ylabel("$p$-value")

    if idx != len(exogenous_instrument_list) - 1:
        axes[idx].set_xticklabels([])
    axes[idx].hlines(
        y=0.05,
        xmin=xrange[0],
        xmax=xrange[1],
        linestyle="--",
        color="red",
        label="0.05" if idx == 0 else None,
    )
    axes[idx].set_ylim(0, 1)

axes[-1].set_xlabel("$\\beta$")
fig.legend(bbox_to_anchor=(0.78, 0.5), loc="center left")

print(table.to_markdown())
print(table.to_latex())
fig.savefig(FIGURES_PATH / "card_2.pdf", bbox_inches="tight")
