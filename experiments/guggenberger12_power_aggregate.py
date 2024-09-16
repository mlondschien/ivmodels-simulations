import json

import click
import matplotlib.pyplot as plt
import numpy as np
from guggenberger12_power_collect import betas

from ivmodels_simulations.constants import DATA_PATH, FIGURES_PATH

input = DATA_PATH / "guggenberger12_power"
figures = FIGURES_PATH / "guggenberger12_power"
figures.mkdir(parents=True, exist_ok=True)

# https://personal.sron.nl/~pault/#sec:qualitative
COLORS = {
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
    "black": "#000000",
    "indigo": "#332288",
    "teal": "#44AA99",
}

COLOR_MAPPING = {
    "AR": COLORS["indigo"],
    "AR (GKM)": COLORS["yellow"],
    "CLR": COLORS["green"],
    "LM (ours)": COLORS["black"],
    "LM (LIML)": COLORS["grey"],
    "LR": COLORS["cyan"],
    "Wald (LIML)": COLORS["red"],
    "Wald (TSLS)": COLORS["blue"],
}

LINESTYLES_MAPPING = {
    "AR": "-",
    "AR (GKM)": (0, (5, 5)),  # loosely dashed
    "CLR": "dotted",
    "LM (ours)": "-",
    "LM (LIML)": (0, (5, 5)),
    "LR": "-",
    "Wald (LIML)": "-",
    "Wald (TSLS)": "-",
}

TESTS = [
    "AR",
    "AR (GKM)",
    "CLR",
    "LM (ours)",
    "LM (LIML)",
    "LR",
    "Wald (LIML)",
    "Wald (TSLS)",
]


@click.command()
@click.option("--n", default=1000)
@click.option("--k", default=10)
def main(n, k):
    with open(input / f"guggenberger_12_power_n={n}_k={k}.json", "r") as f:
        p_values = json.load(f)

    alphas = [0.05, 0.01]
    fig, axes = plt.subplots(nrows=len(alphas), ncols=1, figsize=(10, 5))
    fig.tight_layout(rect=[0.1, 0.01, 0.8, 0.98])
    if len(alphas) == 1:
        axes = [axes]

    for alpha, ax in zip(alphas, axes):
        for test_name in TESTS:
            label = "LM (ours)" if test_name == "LM" else test_name
            ax.plot(
                betas,
                (np.array(p_values[test_name]) < alpha).mean(axis=0),
                label=label if alpha == alphas[0] else None,
                color=COLOR_MAPPING[test_name],
                linestyle=LINESTYLES_MAPPING[test_name],
                lw=2,
            )

        ax.hlines(
            y=alpha,
            xmin=np.min(betas),
            xmax=np.max(betas),
            linestyle="--",
            color="red",
            label="level $\\alpha$" if alpha == alphas[0] else None,
        )
        if alpha == alphas[0]:
            ax.set_title(
                "Power of tests for $\\alpha=0.05$ (top) and $\\alpha=0.01$ (bottom)"
            )
        ax.set_ylabel("rejection frequency")

        ax.vlines(
            x=1,
            ymin=0,
            ymax=1,
            linestyle="--",
            color="black",
            label="$\\beta_0=1$" if alpha == alphas[0] else None,
        )

    ax.set_xlabel("$\\beta$")
    fig.legend(bbox_to_anchor=(0.78, 0.5), loc="center left")
    plt.show()
    fig.savefig(
        figures / f"guggenberger12_power_n={n}_k={k}_alphas={alphas}.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        figures / f"guggenberger12_power_n={n}_k={k}_alphas={alphas}.eps",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
