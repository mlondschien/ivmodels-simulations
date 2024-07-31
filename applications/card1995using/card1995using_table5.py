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
age = ["age76", "age762"]

family = ["daded", "momed", "nodaded", "nomomed", "famed", "momdad14", "sinmom14"]
fs = [f"f{i}" for i in range(1, 9)]  # exclude f9 as sum(f1, ..., f9) = 1
family += fs

# Footnote 23 of Card, 1995: This definition of low family background was derived by
# comparing mean education levels of men in the 8 parental education classes used in
# the models in Table 3 and 4. The means show a discrete drop for men from the two
# lowest parental education categories. I therefore combined the two caregories as a
# "low family background" indicator.
df["low_parental_education"] = df["famed"].isin([8, 9])
df["nearc4_x_low_parental_education"] = df["nearc4"] * df["low_parental_education"]

# Card, 1995, Table 5, b: In column 4 the  instruments are interactions of 8 parental
# education class indicators with an indicator for living near a college in 1966.
nearc4_x_fs = [f"nearc4_x_{f}" for f in fs]
df[nearc4_x_fs] = df[fs].to_numpy() * df[["nearc4"]].to_numpy()

tsls = KClass(kappa="tsls", fit_intercept=True)
print("Column (1) of Table 5. Outcome is ed76.")
summary = tsls.summary(
    X=df[exp],
    Z=df[age],
    C=df[["nearc4", "nearc4_x_low_parental_education"] + indicators + family],
    y=df["ed76"],
    feature_names=["nearc4", "nearc4_x_low_parental_education"],
)
print(summary.coefficient_table_)

tsls = KClass(kappa="tsls", fit_intercept=True)
print("Column (2) of Table 5. Outcome is lwage76.")
summary = tsls.summary(
    X=df[exp],
    Z=df[age],
    C=df[["nearc4", "nearc4_x_low_parental_education"] + indicators + family],
    y=df["lwage76"],
    feature_names=["nearc4", "nearc4_x_low_parental_education"],
)
print(summary.coefficient_table_)


exphat = ["exp76hat", "exp762hat"]

print("Column (3) of Table 5. Outcome is lwage76.")
df[exphat] = proj(
    np.hstack(
        [
            df[
                age
                + indicators
                + family
                + ["nearc4", "nearc4_x_low_parental_education"]
            ],
            np.ones((df.shape[0], 1)),
        ]
    ),
    df[exp].to_numpy(),
)
summary = tsls.summary(
    X=df[["ed76"]],
    Z=df[["nearc4_x_low_parental_education"]],
    C=df[["nearc4"] + indicators + family + exphat],
    y=df["lwage76"],
    feature_names=["ed76", "nearc4"],
)
print(summary.coefficient_table_)


print("Column (4) of Table 5. Outcome is lwage76.")
df[exphat] = proj(
    np.hstack(
        [
            df[age + indicators + family + ["nearc4"] + nearc4_x_fs],
            np.ones((df.shape[0], 1)),
        ]
    ),
    df[exp].to_numpy(),
)
summary = tsls.summary(
    X=df[["ed76"]],
    Z=df[["nearc4"] + nearc4_x_fs],
    C=df[["nearc4"] + indicators + family + exphat],
    y=df["lwage76"],
    feature_names=["ed76", "nearc4"],
)
print(summary.coefficient_table_)
