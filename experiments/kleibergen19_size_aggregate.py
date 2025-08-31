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

font = {"size": 12}

matplotlib.rc("font", **font)


@click.command()
@click.option("--n", default=1000)
@click.option("--k", default=100)
@click.option("--n_vars", default=50)
@click.option("--lambda_max", default=100)
@click.option("--cov_type", default="identity")
@click.option("--n_seeds", default=2500)
def main(n, k, n_vars, lambda_max, n_seeds, cov_type):
    lambda_1s = np.linspace(0, lambda_max, n_vars)
    lambda_2s = np.linspace(0, lambda_max, n_vars)

    lambda_1s, lambda_2s = np.meshgrid(lambda_1s, lambda_2s)

    name = f"kleibergen19_size_n={n}_k={k}_n_seeds={n_seeds}_n_vars={n_vars}_lambda_max={lambda_max}_cov_type={cov_type}.h5"
    file = h5py.File(input / name, "r")
    p_values = {}

    tests = [
        "AR",
        "AR (GKM)",
        "CLR",
        "LM",
        # "LM (LIML)",
        # "LR",
        # "Wald (LIML)",
        # "Wald (TSLS)",
    ]

    for test_name in tests:
        key = test_name if test_name != "AR (GKM)" else "AR (Guggenberger)"
        # key = test_name if test_name != "LM" else "LM (ours)"
        p_values[test_name] = file[key]["p_values"][()] / np.iinfo(data_type).max

    plt.rcParams["axes.titley"] = 0.8
    plt.rcParams["axes.titlepad"] = 0
    plt.locator_params(nbins=2)

    fig_width = 1.5 * 7.5  # * 1.1
    fig_height = 1.5 * 4.72 * 5 / 3
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        subplot_kw={"projection": "3d"},
        figsize=(fig_width, fig_height),
    )

    fig.tight_layout(h_pad=-30, w_pad=5, rect=[0, 0, 0.91, 1])

    my_cmap = cmap.Colormap(
        [
            (0.0, "blue"),
            (0.02, "blue"),
            (0.05, "green"),
            (0.065, "yellow"),
            (0.075, "red"),
            (0.1, "red"),
            (1.0, "red"),
        ],
    ).to_mpl()
    my_cmap.N = 1024  # Increase resolution of colormap

    for idx, (ax, test_name) in enumerate(zip(axes.flat, tests)):
        ax.set_title(test_name if test_name != "LM" else "LM (ours)", loc="left")

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

        ax.set_box_aspect([1, 1, 0.45])  # Make 3d plots "wide"

        ax.set_xlabel(r"$\lambda_1$", rotation=0, labelpad=10)
        ax.xaxis.set_rotate_label(False)

        ax.set_ylabel(r"$\lambda_2$", rotation=0, labelpad=10)
        ax.yaxis.set_rotate_label(False)

        ax.set_zlabel("rejection frequency", rotation=90)
        ax.zaxis.set_rotate_label(False)

        ax.set_facecolor("none")  # So background does not cover title of subplot above

        if idx in [0, 1, 2, 3]:
            ax.set_zlim(0, 0.065)
            # ax.set_zticks([0.0, 0.025, 0.05, 0.075])

    norm1 = matplotlib.colors.Normalize(vmin=0, vmax=0.075)
    color_map1 = matplotlib.colors.LinearSegmentedColormap.from_list(
        "cut_my_cmap1", my_cmap(np.linspace(0, 0.08, my_cmap.N))
    )
    cax1 = plt.axes((0.92, 0.35, 0.025, 0.3))
    cbar1 = matplotlib.colorbar.ColorbarBase(
        cax1,
        cmap=color_map1,
        norm=norm1,
    )
    cbar1.set_ticks([0.0, 0.025, 0.05, 0.075])
    cbar1.set_ticklabels([0.0, 0.025, 0.05, 0.075])

    if cov_type == "identity":
        cov = "$\\Omega = \\mathrm{Id}_3$"
    else:
        cov = "$\\Omega$ as in Guggenberger et al. (2012)"

    fig.suptitle(
        f"Empirical maximal rejection frequencies over $\\tau \\in [0, \\pi)$ for $k={k}$ and {cov}",
        y=0.8,
    )
    # plt.show()
    plt.savefig(
        output
        # eps does not support transparency
        / f"figure_kleibergen19_{cov_type}_k{k}_2x2.pdf",
        # custom as 'tight' cuts of left z-axis label
        bbox_inches=matplotlib.transforms.Bbox([[-0.25, 2.65], [11.25, 9.5]]),
    )


if __name__ == "__main__":
    main()
