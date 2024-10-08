from pathlib import Path

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
}

COLOR_CYCLE = {
    "LR": COLORS["cyan"],
    "Wald (LIML)": COLORS["purple"],
    "AR": COLORS["red"],
    "AR (GKM)": COLORS["yellow"],
    "CLR": COLORS["green"],
    "Wald (TSLS)": COLORS["blue"],
    "LM (LIML)": COLORS["grey"],
    "LM": COLORS["black"],
    "LM (ours)": COLORS["black"],
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
DATA_PATH = Path(__file__).parents[1] / "data"
FIGURES_PATH = Path(__file__).parents[1] / "figures"
TABLES_PATH = Path(__file__).parents[1] / "tables"
