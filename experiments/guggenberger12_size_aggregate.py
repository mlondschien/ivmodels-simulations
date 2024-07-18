# python experiments/guggenberger12_size_aggregate.py
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from ivmodels_simulations.constants import DATA_PATH, FIGURES_PATH, TABLES_PATH

DATA_PATH = DATA_PATH / "testing"
figures = FIGURES_PATH / "testing" / "guggenberger12_size"
figures.mkdir(parents=True, exist_ok=True)
tables = TABLES_PATH / "testing" / "guggenberger12_size"
tables.mkdir(parents=True, exist_ok=True)


@click.command()
@click.option("--n", default=1000)
def main(n):

    alphas = [0.01, 0.05]

    ks = [10, 20, 30]
    tests = [
        "AR",
        "AR (GKM)",
        "CLR",
        "LM (ours)",
        "LM (LIML)",
        "LR",
        "Wald (LIML)",
        "Wald (TSLS)",
    ]

    p_values = pd.read_csv(
        DATA_PATH / "guggenberger12_size" / f"guggenberger12_p_values_n={n}.csv",
        header=[0, 1],
        index_col=None,
    )

    def formatter(x):
        if x >= 10:
            return "{:0.0f}\\%".format(x)
        else:
            return "{:0.1f}\\%".format(x)

    type_1_error_table = pd.DataFrame(
        index=tests, columns=[(alpha, k) for alpha in alphas for k in ks]
    )
    for alpha in alphas:
        for (test_name, k), value in p_values.items():
            if int(k) not in ks:
                continue
            type_1_error_table.loc[test_name][(alpha, int(k))] = np.mean(value < alpha)

    with open(tables / f"guggenberger12_type_1_error_n={n}.txt", "w+") as f:
        formatters = [formatter for _ in ks for _ in alphas]
        f.write((100 * type_1_error_table).to_latex(formatters=formatters))

    print(100 * type_1_error_table)

    fig_width = 1.5 * 6.3
    fig_height = 1.5 * 4.725

    n_seeds = len(p_values)

    fig, axes = plt.subplots(
        nrows=len(tests), ncols=3, figsize=(fig_width, fig_height * 5 / 3)
    )
    fig.suptitle(
        "QQ-plots of $p$-values under the null hypothesis gainst the uniform distribution",
        y=0.94,
    )
    for k_idx, k in enumerate([10, 20, 30]):
        for test_idx, test_name in enumerate(tests):
            values = p_values[(test_name, str(k))]
            if len(values) > 100:
                values = np.sort(values)[(n_seeds // 200) :: (n_seeds // 100)]

            scipy.stats.probplot(
                values,
                dist=scipy.stats.uniform(),
                plot=axes[test_idx, k_idx],
                fit=False,
            )
            axes[test_idx, k_idx].plot([0, 1], [0, 1], color="black")

            if test_idx == 0:
                axes[test_idx, k_idx].set_title(f"k={k}")
            else:
                axes[test_idx, k_idx].set_title("")

            axes[test_idx, k_idx].set_ylabel("Emp. quantiles")

    title_font_size = axes[0, 0].title.get_fontsize()

    for test_idx, test_name in enumerate(tests):
        axes[test_idx, 0].annotate(
            test_name,
            xy=(0, 0.5),
            xytext=(-axes[test_idx, 0].yaxis.labelpad - 5, 0),
            xycoords=axes[test_idx, 0].yaxis.label,
            rotation=0,
            ha="right",
            va="center",
            textcoords="offset points",
            fontsize=title_font_size,
        )

    for ax in fig.get_axes():
        ax.label_outer()

    fig.tight_layout(rect=[0.0, 0.05, 1, 0.95])
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    fig.savefig(
        figures / f"guggenberger12_qqplots_n={n}.eps", format="eps", bbox_inches="tight"
    )
    fig.savefig(
        figures / f"guggenberger12_qqplots_n={n}.pdf", format="pdf", bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
