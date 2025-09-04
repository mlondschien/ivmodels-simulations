from pathlib import Path

import click
import cmap
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from kleibergen19_size_collect import data_type

from ivmodels_simulations.constants import FIGURES_PATH

output = FIGURES_PATH / "clr"
input = Path("/cluster/work/math/lmalte/ivmodels/")
output.mkdir(parents=True, exist_ok=True)

font = {"size": 12}
matplotlib.rc("font", **font)


@click.command()
@click.option("--n", default=1000)
@click.option("--k", default=5)
@click.option("--m", default=2)
@click.option("--n_vars", default=21)
@click.option("--lambda_max", default=100)
@click.option("--n_seeds", default=20_000)
@click.option("--lambda_1", default=10)
def main(n, k, m, n_vars, lambda_max, n_seeds, lambda_1):

    plt.rcParams["axes.titley"] = 0.75
    plt.rcParams["axes.titlepad"] = 0
    plt.locator_params(nbins=2)

    fig_width = 1.5 * 7.5  # * 1.1
    fig_height = 1.5 * 4.72
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        subplot_kw={"projection": "3d"},
        figsize=(fig_width, fig_height),
    )

    fig.tight_layout(h_pad=-5, w_pad=0, rect=[0, 0, 0.96, 1])

    lambda_2s = np.geomspace(1, lambda_max, n_vars)

    # https://sronpersonalpages.nl/~pault/ SEQUENTIAL COLOUR SCHEMES Incandescent
    my_cmap = cmap.Colormap(
        [
            (0.0, "C6F7D6"),
            (0.02, "A2F49B"),
            (0.04, "BBE453"),
            (0.06, "D5CE04"),
            (0.08, "E7B503"),
            (0.1, "F19903"),
            (0.12, "F6790B"),
            (0.14, "F94902"),
            (0.16, "E40515"),
            (1.0, "E40515"),
        ],
    ).to_mpl()
    my_cmap.N = 1024  # Increase resolution of colormap

    beta_mesh, lambda_2_mesh = np.meshgrid(np.linspace(-1, 1, 41), np.log10(lambda_2s))
    for row, (m, k) in enumerate([(2, 10), (4, 20)]):
        for col, lambda_2 in enumerate([5, 10]):
            file_name = f"clr_power_n={n}_k={k}_m={m}_n_seeds={n_seeds}_n_vars={n_vars}_lambda_max={lambda_max}_lambda_1{lambda_2}.h5"
            file = h5py.File(input / file_name, "r")

            new = (
                file["CLR (new)"]["p_values"][()] / np.iinfo(data_type).max < 0.05
            ).mean(axis=0)
            old = (
                file["CLR (old)"]["p_values"][()] / np.iinfo(data_type).max < 0.05
            ).mean(axis=0)
            data = new - old

            axes[row, col].set_title(
                f"k={k}, m={m}\n$\\lambda_1$={lambda_2}", loc="left", fontsize=12
            )

            _ = axes[row, col].plot_surface(
                lambda_2_mesh,
                beta_mesh,
                data,
                rstride=1,
                cstride=1,
                cmap=my_cmap,
                linewidth=0.2,
                antialiased=True,
                vmin=0,
                vmax=1,
                alpha=0.8,
                edgecolor="black",
            )
            axes[row, col].set_proj_type("ortho")

            axes[row, col].invert_yaxis()
            # rotate the axes such that 0, 0, 0 is in the front right
            axes[row, col].view_init(elev=20, azim=200)

            axes[row, col].set_box_aspect([1, 1, 0.45])  # Make 3d plots "wide"

            axes[row, col].set_ylabel(r"$\beta_1$", rotation=0, labelpad=9, fontsize=13)
            axes[row, col].set_xlabel(
                r"$\lambda_2$", rotation=0, labelpad=9, fontsize=13
            )

            axes[row, col].xaxis.set_rotate_label(False)
            axes[row, col].yaxis.set_rotate_label(False)

            if col == 0:
                axes[row, col].set_zlabel("power difference", rotation=90)
                axes[row, col].zaxis.set_rotate_label(False)

            # So background does not cover title of subplot above
            axes[row, col].set_facecolor("none")

            axes[row, col].set_xticks([0.0, 1.0, 2.0])
            axes[row, col].set_xticklabels(["1", "10", "100"])

            if row == 0:
                axes[row, col].set_zlim(0.0, 0.08)
                axes[row, col].set_zticks([0.0, 0.02, 0.04, 0.06, 0.08])
            else:
                axes[row, col].set_zlim(0.0, 0.15)
                axes[row, col].set_zticks([0.0, 0.05, 0.1, 0.15])

    norm1 = matplotlib.colors.Normalize(vmin=0, vmax=0.15)
    color_map1 = matplotlib.colors.LinearSegmentedColormap.from_list(
        "cut_my_cmap1", my_cmap(np.linspace(0, 0.15, my_cmap.N))
    )
    cax1 = plt.axes((0.88, 0.35, 0.025, 0.3))
    cbar1 = matplotlib.colorbar.ColorbarBase(
        cax1,
        cmap=color_map1,
        norm=norm1,
    )
    cbar1.set_ticks([0.0, 0.05, 0.1, 0.15])
    cbar1.set_ticklabels([0.0, 0.05, 0.1, 0.15])

    fig.savefig(
        output
        # eps does not support transparency
        / "figure_clr_power.pdf",
        # custom as 'tight' cuts of left z-axis label
        # [left, bottom], [right, top]
        bbox_inches=matplotlib.transforms.Bbox([[0.8, 0.6], [10.65, 6.3]]),
    )


if __name__ == "__main__":
    main()
