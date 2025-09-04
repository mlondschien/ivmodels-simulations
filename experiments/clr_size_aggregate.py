import click
import cmap
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from kleibergen19_size_collect import data_type

from ivmodels_simulations.constants import DATA_PATH, FIGURES_PATH

output = FIGURES_PATH / "clr"
input = DATA_PATH / "clr"
output.mkdir(parents=True, exist_ok=True)

font = {"size": 12}
matplotlib.rc("font", **font)


@click.command()
@click.option("--n", default=1000)
@click.option("--n_vars", default=21)
@click.option("--lambda_max", default=100)
def main(n, n_vars, lambda_max):

    plt.rcParams["axes.titley"] = 0.7
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

    lambda_1s = np.geomspace(1, lambda_max, n_vars)
    lambda_2s = np.geomspace(1, lambda_max, n_vars)

    lambda_1s, lambda_2s = np.meshgrid(lambda_1s, lambda_2s)

    my_cmap = cmap.Colormap(
        [
            (0.0, "blue"),
            (0.03, "blue"),
            (0.045, "green"),
            (0.05, "green"),
            (0.06, "yellow"),
            (0.07, "red"),
            (0.1, "red"),
            (1.0, "red"),
        ],
    ).to_mpl()

    my_cmap.N = 1024  # Increase resolution of colormap

    for row, (n_seeds, m, k) in enumerate([[50000, 2, 10], [50000, 4, 20]]):
        name = f"clr_size_n={n}_k={k}_m={m}_n_seeds={n_seeds}_n_vars={n_vars}_lambda_max={lambda_max}.h5"
        file = h5py.File(input / name, "r")
        p_values = {}

        tests = [ "CLR (old)", "CLR (new)"]

        for test_name in tests:
            p_values[test_name] = (
                file[test_name]["p_values"][()] / np.iinfo(data_type).max
            )

        for idx, test_name in enumerate(tests):
            data = file[test_name]["p_values"][()] / np.iinfo(data_type).max
            data = (data < 0.05).mean(axis=0)
            axes[row, idx].set_title(
                f"{test_name}\nk={k}\nm={m}", loc="left", fontsize=12
            )

            _ = axes[row, idx].plot_surface(
                np.log10(lambda_2s),
                np.log10(lambda_1s),
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
            axes[row, idx].set_proj_type("ortho")

            # rotate the axes such that 0, 0, 0 is in the front right
            axes[row, idx].view_init(elev=20, azim=200)
            axes[row, idx].set_box_aspect([1, 1, 0.45])  # Make 3d plots "wide"

            axes[row, idx].set_ylabel(
                r"$\lambda_1$", rotation=0, labelpad=9, fontsize=13
            )
            axes[row, idx].set_xlabel(
                r"$\lambda_2$", rotation=0, labelpad=9, fontsize=13
            )

            axes[row, idx].xaxis.set_rotate_label(False)
            axes[row, idx].yaxis.set_rotate_label(False)

            if idx == 0:
                axes[row, idx].set_zlabel("rejection frequency", rotation=90)
                axes[row, idx].zaxis.set_rotate_label(False)

            # So background does not cover title of subplot above
            axes[row, idx].set_facecolor("none")

            axes[row, idx].set_zlim(0.02, 0.06)
            axes[row, idx].set_xticks([0.0, 1.0, 2.0])
            axes[row, idx].set_xticklabels(["1", "10", "100"])
            axes[row, idx].set_yticks([0.0, 1.0, 2.0])
            axes[row, idx].set_yticklabels(["1", "10", "100"])

            axes[row, idx].set_zticks([0.02, 0.03, 0.04, 0.05, 0.06])

    norm1 = matplotlib.colors.Normalize(vmin=0.02, vmax=0.07)
    color_map1 = matplotlib.colors.LinearSegmentedColormap.from_list(
        "cut_my_cmap1", my_cmap(np.linspace(0.02, 0.07, my_cmap.N))
    )
    cax1 = plt.axes((0.88, 0.35, 0.025, 0.3))
    cbar1 = matplotlib.colorbar.ColorbarBase(
        cax1,
        cmap=color_map1,
        norm=norm1,
    )
    cbar1.set_ticks([0.03, 0.05, 0.07])
    cbar1.set_ticklabels([0.03, 0.05, 0.07])

    plt.savefig(
        output
        # eps does not support transparency
        / "figure_clr_size.pdf",
        # custom as 'tight' cuts of left z-axis label
        # [left, bottom], [right, top]
        bbox_inches=matplotlib.transforms.Bbox([[0.8, 0.6], [10.65, 6.3]]),
    )


if __name__ == "__main__":
    main()
