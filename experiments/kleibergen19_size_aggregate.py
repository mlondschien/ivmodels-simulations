import click
import cmap
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from kleibergen19_size_collect import data_type

from ivmodels_simulations.constants import DATA_PATH, FIGURES_PATH

output = FIGURES_PATH / "kleibergen19_size"
input = DATA_PATH / "kleibergen19_size"
output.mkdir(parents=True, exist_ok=True)


@click.command()
@click.option("--n", default=1000)
@click.option("--k", default=100)
@click.option("--n_vars", default=20)
@click.option("--lambda_max", default=20)
@click.option("--cov_type", default="identity")
@click.option("--n_seeds", default=1000)
def main(n, k, n_vars, lambda_max, n_seeds, cov_type):
    lambda_1s = np.linspace(0, lambda_max, n_vars)
    lambda_2s = np.linspace(0, lambda_max, n_vars)

    lambda_1s, lambda_2s = np.meshgrid(lambda_1s, lambda_2s)

    name = f"kleibergen19_size_n={n}_k={k}_n_seeds={n_seeds}_n_vars={n_vars}_lambda_max={lambda_max}_cov_type={cov_type}.h5"
    file = h5py.File(input / name, "r")
    p_values = {}

    tests = {
        "AR": "AR",
        "AR (Guggenberger)": "AR (GKM)",
        "CLR": "CLR",
        "LM": "LM (us)",
        "LM (LIML)": "LM (LIML)",
        "LR": "LR",
        "Wald (LIML)": "Wald (LIML)",
        "Wald (TSLS)": "Wald (TSLS)",
    }

    for test_name in tests:
        p_values[test_name] = file[test_name]["p_values"][()] / np.iinfo(data_type).max

    plt.rcParams["axes.titley"] = 0.82
    plt.rcParams["axes.titlepad"] = 0
    plt.locator_params(nbins=4)

    fig_width = 1.5 * 7.5
    fig_height = 1.5 * 4.725 * 5 / 3
    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        subplot_kw={"projection": "3d"},
        figsize=(fig_width, fig_height),
    )

    fig.tight_layout(h_pad=-6, rect=[0, 0, 0.95, 1])

    my_cmap = cmap.Colormap(
        [
            (0.0, "blue"),
            (0.02, "blue"),
            (0.05, "green"),
            (0.07, "yellow"),
            (0.1, "red"),
            (1.0, "red"),
        ]
    ).to_mpl()

    for idx, (ax, test_name, title) in enumerate(
        zip(axes.flat, tests.keys(), tests.values())
    ):
        ax.set_title(title, loc="left")

        data = (p_values[test_name] < 0.05).mean(axis=0).max(axis=0)

        _ = ax.plot_surface(
            lambda_1s,
            lambda_2s,
            data,
            rstride=5,
            cstride=5,
            cmap=my_cmap,
            linewidth=0.2,
            antialiased=True,
            vmin=0,
            vmax=1,
            alpha=0.8,
            edgecolor="black",
        )
        ax.set_proj_type("ortho")

        # rotate the axes such that 0, 0, 0 is in the front right
        ax.view_init(elev=20, azim=200)

        ax.set_box_aspect([1, 1, 0.5])  # Make 3d plots "wide"

        ax.set_xlabel(r"$\lambda_1$", rotation=0)
        ax.xaxis.set_rotate_label(False)

        ax.set_ylabel(r"$\lambda_2$", rotation=0)
        ax.yaxis.set_rotate_label(False)

        ax.set_zlabel("rejection frequency", rotation=90)
        ax.zaxis.set_rotate_label(False)

        ax.set_facecolor("none")  # So background does not cover title of subplot above

    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.1)
    colors = my_cmap(np.linspace(0, 0.1, my_cmap.N))
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list(
        "cut_my_cmap", colors
    )

    # create some axes to put the colorbar to
    cax = plt.axes((0.85, 0.35, 0.025, 0.3))
    cbar = matplotlib.colorbar.ColorbarBase(
        cax,
        cmap=color_map,
        norm=norm,
    )

    cbar.set_ticks([0.0, 0.025, 0.05, 0.075, 0.1])
    cbar.set_ticklabels([0.0, 0.025, 0.05, 0.075, 0.1])

    plt.savefig(
        output
        / f"figure_kleibergen19_{cov_type}_k{k}.pdf",  # eps does not support transparency
        bbox_inches="tight",
        pad_inches=0.25,  # need some pad for z-axis label left
    )


if __name__ == "__main__":
    main()
