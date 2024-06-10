import numpy as np
import click
import h5py
from kleibergen19_size_collect import tests
import matplotlib.pyplot as plt
from matplotlib import cm
from ivmodels_simulations.constants import DATA_PATH

output = DATA_PATH / "kleibergen19_size"

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
    file = h5py.File(output / name, "r")
    p_values = {}

    tests = [
        "LM (LIML)",
        # "Wald (TSLS)",
        "Wald (LIML)",
        "LR",
        "LM",
        "AR",
        "CLR",
        # "AR (Guggenberger)",
    ]


    for test_name in tests:
        p_values[test_name] = file[test_name]["p_values"][()]


    max_rejection = np.max([np.max((v < 0.05).mean(axis=0).max(axis=0)) for v in p_values.values()])

    fig, axes = plt.subplots(nrows=3, ncols=2, subplot_kw={"projection": "3d"})

    surfaces = []

    for idx, (ax, test_name) in enumerate(zip(axes.flat, tests)):
        # Plot the surface.
        surfaces.append(ax.plot_surface(lambda_1s, lambda_2s, (p_values[test_name] < 0.05).mean(axis=0).max(axis=0), cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, vmin=0, vmax=max_rejection))
        ax.set_title(test_name)

        # Customize the z axis.
        ax.set_zlim(0, max_rejection * 1.1)

        # rotate the axes such that 0, 0, 0 is in the front
        ax.view_init(elev=20, azim=45)

    fig.colorbar(surfaces[0], ax=axes.ravel().tolist(), shrink=0.5, aspect=5)

    fig.tight_layout()
    plt.savefig(output / f"kleibergen19_size_n={n}_k={k}_n_seeds={n_seeds}_n_vars={n_vars}_lambda_max={lambda_max}_cov_type={cov_type}.png")

if __name__ == "__main__":
    main()