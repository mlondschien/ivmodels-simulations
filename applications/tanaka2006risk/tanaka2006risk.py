import re
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
df = pd.read_stata(data_dir / "20060431_risk.dta")
df["rainfallxheadnowork"] = df["rainfall"] * df["headnowork"]

outcome = "vfctnc"
exogenous_names = ["chinese", "edu", "market", "south", "age"]
gender = ["gender"]
endogenous_names = ["nmlrlincome", "mnincome"]
instrument_names = ["rainfall", "headnowork"]
instrument_names_2 = ["rainfall", "headnowork", "rainfallxheadnowork"]
income = ["nmlrlincome", "mnincome"]

alphas = [0.2, 0.05]

# X, W, C, D, Z
x_w_c_d_z_lists = [
    [
        (
            ["mnincome"],
            ["nmlrlincome"],
            exogenous_names + gender,
            [],
            instrument_names,
        ),
        (
            ["nmlrlincome"],
            ["mnincome"],
            exogenous_names + gender,
            [],
            instrument_names,
        ),
        ([], income, exogenous_names, gender, instrument_names),
    ],
    [
        (
            ["mnincome"],
            ["nmlrlincome"],
            exogenous_names + gender,
            [],
            instrument_names + ["rainfallxheadnowork"],
        ),
        (
            ["nmlrlincome"],
            ["mnincome"],
            exogenous_names + gender,
            [],
            instrument_names + ["rainfallxheadnowork"],
        ),
        (
            [],
            income,
            exogenous_names,
            gender,
            instrument_names + ["rainfallxheadnowork"],
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
    (
        "CLR",
        conditional_likelihood_ratio_test,
        inverse_conditional_likelihood_ratio_test,
    ),
]
index = [("estimate (OLS)", ""), ("estimate (TSLS)", ""), ("estimate (LIML)", "")]

for test_name, _, _ in tests:
    index += [(test_name, "stat. | $p$-value")]
    for alpha in alphas:
        index += [(test_name, f"{100 * (1 - alpha):.0f}\\% conf. set")]
index += [("rank", "stat. | $p$-value"), ("$J_\\liml$", "stat. | $p$-value")]

for ldx, x_w_c_d_z_list in enumerate(x_w_c_d_z_lists):
    print(ldx)
    table_index = pd.MultiIndex.from_tuples(index, names=["first", "second"])
    columns = pd.MultiIndex.from_tuples(
        [(idx, jdx) for idx in range(len(x_w_c_d_z_list)) for jdx in range(2)]
    )
    table = pd.DataFrame(columns=columns, index=table_index)

    for idx, (x_vars, w_vars, c_vars, d_vars, z_vars) in enumerate(x_w_c_d_z_list):
        X = df[x_vars]
        W = df[w_vars]
        XW = df[x_vars + w_vars]
        y = df[outcome]
        C = df[c_vars]
        D = df[d_vars]
        CD = df[c_vars + d_vars]
        Z = df[z_vars]

        cdx = 0 if len(d_vars) == 0 else -1

        for est in "OLS", "TSLS", "LIML":
            model = KClass(kappa=est, fit_intercept=True)
            model.fit(X=XW, y=y, Z=Z, C=CD)

            table.loc[(f"estimate ({est})", ""), (idx, 0)] = f"{model.coef_[cdx]:.3f}"
            table.loc[(f"estimate ({est})", ""), (idx, 1)] = (
                f"({np.abs(model.coef_[cdx]) / np.sqrt(wald_test(X=X, W=W, y=y, Z=Z, C=C, D=D, beta=np.zeros(1), estimator=est)[0]):.3f})"
            )

        for test_name, test, inverse_test in tests:
            stat, p_value = test(X=X, W=W, y=y, Z=Z, C=C, D=D, beta=np.zeros(1))
            table.loc[(test_name, "stat. | $p$-value"), (idx, 0)] = f"{stat:.3f}"
            table.loc[(test_name, "stat. | $p$-value"), (idx, 1)] = f"{p_value:.3f}"

            for alpha in alphas:
                conf_set = inverse_test(X=X, W=W, y=y, Z=Z, C=C, D=D, alpha=alpha)
                if isinstance(conf_set, Quadric):
                    conf_set = ConfidenceSet.from_quadric(conf_set)
                table.loc[
                    (test_name, f"{100 * (1 - alpha):.0f}\\% conf. set"), (idx, 0)
                ] = f"{conf_set:.3f}"

        stat, p_value = rank_test(X=XW, Z=Z, C=CD)
        table.loc[("rank", "stat. | $p$-value"), (idx, 0)] = f"{stat:.3f}"
        table.loc[("rank", "stat. | $p$-value"), (idx, 1)] = f"{p_value:.3f}"
        stat, p_value = j_test(X=XW, y=y, Z=Z, C=CD, estimator="liml")
        table.loc[("$J_\\liml$", "stat. | $p$-value"), (idx, 0)] = f"{stat:.3f}"
        table.loc[("$J_\\liml$", "stat. | $p$-value"), (idx, 1)] = f"{p_value:.3f}"

    table.index = table.index.map(" ".join)
    latex = table.to_latex(multirow=False)
    latex = re.sub("\\& \\[", "& \\\\multicolumn{2}{r}{[", latex)
    latex = re.sub("] & NaN", "]}", latex)

    print(table.to_markdown())
    print(latex)
