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

college_prox = ["nearc4a", "nearc4b", "nearc2"]  # college proximity

family = ["daded", "momed", "nodaded", "nomomed", "famed", "momdad14", "sinmom14"]
fs = [f"f{i}" for i in range(1, 9)]  # exclude f9 as sum(f1, ..., f9) = 1
family += fs
exogenous = indicators + family

df["low_par_ed"] = df["famed"].isin([8, 9])  # low parental education
college_prox_x_low_par_ed = [f"{v}_x_low_par_ed" for v in college_prox]
df[college_prox_x_low_par_ed] = df[college_prox] * df[["low_par_ed"]].to_numpy()

df["black_ed76"] = df["black"] * df["ed76"]  # Interaction of race (black) w/ education
df["black_x_nearc4"] = df["black"] * df["nearc4"]  # Interaction of race w/ college prox

black = df[["black"]].to_numpy()
black_x_college_prox = [f"black_x_{v}" for v in college_prox]
df[black_x_college_prox] = df[college_prox] * black

black_x_college_prox_x_low_par_ed = [f"black_x_{v}" for v in college_prox_x_low_par_ed]
df[black_x_college_prox_x_low_par_ed] = df[college_prox_x_low_par_ed] * black

x_w_z_lists = [
    [
        (["ed76"], ["exp76", "exp762"], age + ["nearc4"]),
        (["ed76"], ["exp76", "exp762"], age + college_prox),
        (["ed76"], ["exp76", "exp762"], age + college_prox + college_prox_x_low_par_ed),
    ],
    [
        (
            ["ed76"],
            ["exp76", "exp762", "black_ed76"],
            age + ["nearc4", "black_x_nearc4"],
        ),
        (
            ["ed76"],
            ["exp76", "exp762", "black_ed76"],
            age + college_prox + black_x_college_prox,
        ),
        (
            ["ed76"],
            ["exp76", "exp762", "black_ed76"],
            age
            + college_prox
            + black_x_college_prox
            + college_prox_x_low_par_ed
            + black_x_college_prox_x_low_par_ed,
        ),
    ],
    [
        (
            ["black_ed76"],
            ["exp76", "exp762", "ed76"],
            age + ["nearc4", "black_x_nearc4"],
        ),
        (
            ["black_ed76"],
            ["exp76", "exp762", "ed76"],
            age + college_prox + black_x_college_prox,
        ),
        (
            ["black_ed76"],
            ["exp76", "exp762", "ed76"],
            age
            + college_prox
            + black_x_college_prox
            + college_prox_x_low_par_ed
            + black_x_college_prox_x_low_par_ed,
        ),
    ],
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

xrange = (-0.5, 0.5)
specifications = {0: "(i)", 1: "(ii)", 2: "(iii)", 3: "(iv)", 4: "(v)"}


for ldx, x_w_z_list in enumerate(x_w_z_lists):
    print(ldx)
    table_index = pd.MultiIndex.from_tuples(index, names=["first", "second"])
    columns = pd.MultiIndex.from_tuples(
        [(idx, jdx) for idx in range(len(x_w_z_list)) for jdx in range(2)]
    )
    table = pd.DataFrame(columns=columns, index=table_index)

    fig, axes = plt.subplots(len(x_w_z_list), figsize=(10, 7))
    fig.tight_layout(rect=[0.1, 0.01, 0.8, 0.98])

    for idx, (x_vars, w_vars, z_vars) in enumerate(x_w_z_list):
        print(f"specification {specifications[idx]}, k={len(z_vars)}")
        X = df[x_vars]
        W = df[w_vars]
        XW = df[x_vars + w_vars]
        y = df["lwage76"]
        C = df[indicators + family]
        Z = df[z_vars]

        for est in "TSLS", "LIML":
            model = KClass(kappa=est, fit_intercept=True)
            model.fit(X=XW, y=y, Z=Z, C=C)

            wald = wald_test(X=X, W=W, y=y, Z=Z, C=C, beta=np.zeros(1), estimator=est)
            std_error = np.abs(model.coef_[0]) / np.sqrt(wald[0])

            table.loc[(f"estimate ({est})", ""), (idx, 0)] = f"{model.coef_[0]:.3g}"
            table.loc[(f"estimate ({est})", ""), (idx, 1)] = f"({std_error:.3g})"

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

        axes[idx].set_title(
            f"specification {specifications[idx]}, k={Z.shape[1]}",
            fontsize=12,
        )

        axes[idx].set_ylabel("$p$-value")

        if idx != len(x_w_z_list) - 1:
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
    fig.savefig(FIGURES_PATH / f"card_{ldx}.pdf", bbox_inches="tight")
    print(table.to_markdown())
    print(table.to_latex())
