# Replicate Card (1995) Table 3
import numpy as np
from ivmodels import KClass
from ivmodels.utils import proj

from ivmodels_simulations.load import load_card1995using

df = load_card1995using()

# All models include a black indicator, indicators for southern residence and residence
# in an SMSA in 1976, indicators for region in 1966 and living in an SMSA in 1966, as
# well as experience and experience squared.
indicators = ["black", "smsa66r", "smsa76r", "reg76r"]
# exclude reg669, as sum(reg661, ..., reg669) = 1
indicators += [f"reg66{i}" for i in range(1, 9)]
exp = ["exp76", "exp762"]

family = ["daded", "momed", "nodaded", "nomomed", "famed", "momdad14", "sinmom14"]
family += [f"f{i}" for i in range(1, 8)]  # exclude f8 as sum(f1, ..., f8) = 1

test = "wald"
# Replicate Table 3 from Card (1995)
# row 1: OLS estimates of the effect of college proximity on education and log wages
ols = KClass(kappa="ols", fit_intercept=True)
print("Column (1) row 1 of Table 3. Outcome is ed76, no family variables.")
summary = ols.summary(
    test=test,
    X=df[["nearc4"] + indicators + exp],
    y=df["ed76"],
    feature_names=["nearc4"],
)
print(summary.coefficient_table_)

print("\nColumn (2) row 1 of Table 3. Outcome is ed76, with family variables.")
summary = ols.summary(
    test=test,
    X=df[["nearc4"] + indicators + exp + family],
    y=df["ed76"],
    feature_names=["nearc4"],
)
print(summary.coefficient_table_)

print("\nColumn (3) row 1 of Table 3. Outcome is lwage76, no family variables.")
summary = ols.summary(
    test=test,
    X=df[["nearc4"] + indicators + exp],
    y=df["lwage76"],
    feature_names=["nearc4"],
)
print(summary.coefficient_table_)

print("\nColumn (4) row 1 of Table 3. Outcome is lwage76, with background variables.")
summary = ols.summary(
    test=test,
    X=df[["nearc4"] + indicators + exp + family],
    y=df["lwage76"],
    feature_names=["nearc4"],
)
print(summary.coefficient_table_)

# row 2: 2SLS estimates of the effect of education and log wages
tsls = KClass(kappa="tsls", fit_intercept=True)
print(
    "\nColumn (5) row 2 of Table 3. Outcome is lwage76, instrumented by college "
    "proximity (nearc4), no family variables."
)
summary = tsls.summary(
    test=test,
    X=df[["ed76"]],
    y=df["lwage76"],
    Z=df[["nearc4"]],
    C=df[indicators + exp],
    feature_names=["ed76"],
)
print(summary.coefficient_table_)

print(
    "\nColumn (6) row 2 of Table 3. Outcome is lwage76, instrumented by college "
    "proximity (nearc4), with family variables."
)
summary = tsls.summary(
    test=test,
    X=df[["ed76"]],
    y=df["lwage76"],
    Z=df[["nearc4"]],
    C=df[indicators + exp + family],
    feature_names=["ed76"],
)
print(summary.coefficient_table_)

# row 4: 2SLS estimates of the effect of college proximity on education and log wages,
# with experience and experience squared instrumented by age and age squared
print(
    "Column (1) row 4 of Table 3. Outcome is ed76, no family variables. Variables "
    "exp76 and exp762 instrumented with age76 and age762."
)
summary = tsls.summary(
    test=test,
    X=df[exp],
    y=df["ed76"],
    C=df[["nearc4"] + indicators],
    Z=df[["age76", "age762"]],
    feature_names=["nearc4"],
)
print(summary.coefficient_table_)

print(
    "\nColumn (2) row 4 of Table 3. Outcome is ed76, with family variables. Variables "
    "exp76 and exp762 instrumented with age76 and age762."
)
summary = tsls.summary(
    test=test,
    X=df[exp],
    y=df["ed76"],
    C=df[["nearc4"] + indicators + family],
    Z=df[["age76", "age762"]],
    feature_names=["nearc4"],
)
print(summary.coefficient_table_)

print(
    "\nColumn (3) row 4 of Table 3. Outcome is lwage76, no family variables. Variables "
    "exp76 and exp762 instrumented with age76 and age762."
)
summary = tsls.summary(
    test=test,
    X=df[exp],
    y=df["lwage76"],
    C=df[["nearc4"] + indicators],
    Z=df[["age76", "age762"]],
    feature_names=["nearc4"],
)
print(summary.coefficient_table_)


print(
    "\nColumn (4) row 4 of Table 3. Outcome is lwage76, with family variables. Variables "
    "exp76 and exp762 instrumented with age76 and age762."
)
summary = tsls.summary(
    test=test,
    X=df[exp],
    y=df["lwage76"],
    C=df[["nearc4"] + family + indicators],
    Z=df[["age76", "age762"]],
    feature_names=["nearc4"],
)
print(summary.coefficient_table_)


# row 5: 2SLS estimates of the effect of college proximity on education and log wages,
# with potential experience and its square instrumented by age and age squared
exphat = ["exp76hat", "exp762hat"]
age = ["age76", "age762"]
tsls = KClass(kappa="tsls", fit_intercept=True)
print(
    "\nColumn (5) row 5 of Table 3. Outcome is lwage76, instrumented by college "
    "proximity (nearc4), no family variables."
)
df[exphat] = proj(
    np.hstack([df[age + indicators + ["nearc4"]], np.ones((df.shape[0], 1))]),
    df[exp].to_numpy(),
)
summary = tsls.summary(
    test=test,
    X=df[["ed76"]],
    y=df["lwage76"],
    Z=df[["nearc4"]],
    C=df[indicators + exphat],
    feature_names=["ed76"],
)
print(summary.coefficient_table_)

print(
    "\nColumn (6) row 5 of Table 3. Outcome is lwage76, instrumented by college "
    "proximity (nearc4), with family variables."
)
df[exphat] = proj(
    np.hstack([df[age + indicators + family + ["nearc4"]], np.ones((df.shape[0], 1))]),
    df[exp].to_numpy(),
)
summary = tsls.summary(
    test=test,
    X=df[["ed76"]],
    y=df["lwage76"],
    Z=df[["nearc4"]],
    C=df[indicators + exphat + family],
    feature_names=["ed76"],
)
print(summary.coefficient_table_)
