from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests

CACHE_DIR = Path(__file__).parents[2] / "applications_data" / "card1995using"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_card1995using(cache=True):
    cache_file = CACHE_DIR / "card1995using.parquet"
    if cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
    else:
        url = "https://davidcard.berkeley.edu/data_sets/proximity.zip"
        content = requests.get(url).content

        # From code_bk.txt in the zip file
        colspec = {
            "id": (1, 5),  # sequential id runs from 1 to 5225
            "nearc2": (7, 7),  # grew up near 2-yr college
            "nearc4": (10, 10),  # grew up near 4-yr college
            "nearc4a": (12, 13),  # grew up near 4-yr public college
            "nearc4b": (15, 16),  # grew up near 4-yr priv college
            "ed76": (18, 19),  # educ in 1976
            "ed66": (21, 22),  # educ in 1966
            "age76": (24, 25),  # age in 1976
            "daded": (27, 31),  # dads education missing=avg
            "nodaded": (33, 33),  # 1 if dad ed imputed
            "momed": (35, 39),  # moms education
            "nomomed": (41, 41),  # 1 if mom ed imputed
            "weight": (43, 54),  # nls weight for 1976 cross-section
            "momdad14": (56, 56),  # 1 if live with mom and dad age 14
            "sinmom14": (58, 58),  # lived with single mom age 14
            "step14": (60, 60),  # lived step parent age 14
            "reg661": (62, 62),  # dummy for region=1 in 1966
            "reg662": (64, 64),  # dummy for region=2 in 1966
            "reg663": (66, 66),  # dummy for region=3 in 1966
            "reg664": (68, 68),
            "reg665": (70, 70),
            "reg666": (72, 72),
            "reg667": (74, 74),
            "reg668": (76, 76),
            "reg669": (78, 78),  # dummy for region=9 in 1966
            "south66": (80, 80),  # lived in south in 1966
            "work76": (82, 82),  # worked in 1976
            "work78": (84, 84),  # worked in 1978
            "lwage76": (86, 97),  # log wage (outliers trimmed) 1976
            "lwage78": (99, 110),  # log wage in 1978 outliers trimmed
            "famed": (112, 112),  # mom-dad education class 1-9
            "black": (114, 114),  # 1 if black
            "smsa76r": (116, 116),  # in smsa in 1976
            "smsa78r": (118, 118),  # in smsa in 1978
            "reg76r": (120, 120),  # in south in 1976
            "reg78r": (122, 122),  # in south in 1978
            "reg80r": (124, 124),  # in south in 1980
            "smsa66r": (126, 126),  # in smsa in 1966
            "wage76": (128, 132),  # raw wage cents per hour 1976
            "wage78": (134, 138),
            "wage80": (140, 144),
            "noint78": (146, 146),  # 1 if noninterview in 78
            "noint80": (148, 148),
            "enroll76": (150, 150),  # 1 if enrolled in 76
            "enroll78": (152, 152),
            "enroll80": (154, 154),
            "kww": (156, 157),  # the kww score
            "iq": (159, 161),  # a normed iq score
            "marsta76": (163, 163),  # mar status in 1976 1=married, sp. present
            "marsta78": (165, 165),
            "marsta80": (167, 167),
            "libcrd14": (169, 169),  # 1 if lib card in home age 14
        }

        with ZipFile(BytesIO(content)).open("nls.dat") as file:
            df = pd.read_fwf(
                file,
                names=colspec.keys(),
                # pandas expects [from, to[ values, starting at 0
                colspecs=[(f - 1, t) for (f, t) in colspec.values()],
                na_values=".",
            )

        if cache:
            df.to_parquet(cache_file)

    df = df.set_index("id")
    df = df[lambda x: x["lwage76"].notna()]

    # construct potential experience and its square
    df["exp76"] = df["age76"] - df["ed76"] - 6
    df["exp762"] = df["exp76"] ** 2
    df["age762"] = df["age76"] ** 2

    df["f1"] = df["famed"].eq(1).astype("float")  # mom and dad both > 12 yrs ed
    df["f2"] = df["famed"].eq(2).astype("float")  # mom&dad >=12 and not both exactly 12
    df["f3"] = df["famed"].eq(3).astype("float")  # mom=dad=12
    df["f4"] = df["famed"].eq(4).astype("float")  # mom >=12 and dad missing
    df["f5"] = df["famed"].eq(5).astype("float")  # father >=12 and mom not in f1-f4
    df["f6"] = df["famed"].eq(6).astype("float")  # mom>=12 and dad nonmissing
    df["f7"] = df["famed"].eq(7).astype("float")  # mom and dad both >=9
    df["f8"] = df["famed"].eq(8).astype("float")  # mom and dad both nonmissing

    return df
