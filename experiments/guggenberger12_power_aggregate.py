import json

import matplotlib.pyplot as plt
import numpy as np
from guggenberger12_power_collect import n_betas

from ivmodels_simulations.constants import DATA_PATH, FIGURES_PATH

input = DATA_PATH / "guggenberger12_power"
figures = FIGURES_PATH / "guggenberger12_power"
figures.mkdir(parents=True, exist_ok=True)

with open(input / "guggenberger_12_power.json", "r") as f:
    p_values = json.load(f)

betas = np.linspace(0.5, 1.5, n_betas)

alpha = 0.05
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 6))

for test_name in p_values.keys():
    ax.plot(
        betas, (np.array(p_values[test_name]) < alpha).mean(axis=0), label=test_name
    )

ax.hlines(
    y=alpha,
    xmin=np.min(betas),
    xmax=np.max(betas),
    linestyle="--",
    color="red",
    label=f"$\\alpha={alpha}$",
)
ax.vlines(x=1, ymin=0, ymax=1, linestyle="--", color="black", label="$\\beta_0=1$")
ax.legend()

fig.savefig(figures / f"guggenberger12_power_{alpha}.pdf")
fig.savefig(figures / f"guggenberger12_power_{alpha}.eps")
