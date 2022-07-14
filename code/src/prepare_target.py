import pandas as pd
import numpy as np
import glob

def short_date_format(row):
    row[0] = row[0][3:]
    return row


DATA_PATH = "data/ojdamage_rus.xls"
REGION = "Субъект Российской Федерации "
CUR_REGION = "Тамбовская область"
EVENTNAME_COL = "Название явления "
STARTDATE = "Дата начала "
EVENTTYPE = "Град"
MONTH_COL = "Month"
YEAR_COL = "Year"
TARGET = "target"


def get_target(data_path = DATA_PATH, region = CUR_REGION):
    data = pd.read_excel(data_path)
    hail_data = data[data[EVENTNAME_COL] == EVENTTYPE].reset_index().drop(columns="index")[[STARTDATE, REGION]]
    hail_data = hail_data[hail_data[REGION] == region].reset_index().drop(columns="index")
    hail_data[STARTDATE] = hail_data[[STARTDATE]].apply(short_date_format, axis=1)
    hail_data = hail_data.drop_duplicates()
    hail_data[STARTDATE] = pd.to_datetime(hail_data[STARTDATE], format="%m.%Y")  # , dayfirst = True)
    hail_data = hail_data.sort_values(by=[STARTDATE])
    hail_data = hail_data.drop(columns=[REGION])
    hail_data[TARGET] = np.ones(hail_data.shape[0], dtype=int)
    hail_data = hail_data.set_index([STARTDATE])
    idx = pd.date_range(min(hail_data.index), max(hail_data.index), freq='MS')
    hail_data = hail_data.reindex(idx, fill_value=0)
    return hail_data


IMPORTANT_COLS = [
    "STATE",
    "BEGIN_DATE_TIME",
    "END_DATE_TIME",
    "MAGNITUDE",
    "DAMAGE_CROPS",
    "DAMAGE_PROPERTY",
    "BEGIN_LAT",
    "BEGIN_LON",
    "END_LAT",
    "END_LON",
]

#####################
#                   #
# Дописать get_grid #
#                   #
#####################


def round_to25(n: float):
    floor = np.floor(n)
    if abs(n - floor) <= 0.125:
        return floor
    elif abs(n - (floor + 0.125)) <= 0.125:
        return floor + 0.25
    elif abs(n - (floor + 0.5)) <= 0.125:
        return floor + 0.5
    elif abs(n - (floor + 0.75)) <= 0.125:
        return floor + 0.75
    else:
        return floor + 1


def round_coord(row):
    row[0] = round_to25(row[0])
    row[1] = round_to25(row[1])
    return row


def get_grid(dataframe: pd.DataFrame, lat: tuple, long: tuple, year: int):
    leap_year = [2016, 2020]
    num_of_days = 365

    if year in leap_year:
        num_of_days = 366

    long_grid = [-109., -108.75, -108.5, -108.25, -108., -107.75, -107.5, -107.25,
                 -107., -106.75, -106.5, -106.25, -106., -105.75, -105.5, -105.25,
                 -105., -104.75, -104.5, -104.25, -104., -103.75, -103.5, -103.25,
                 -103., -102.75, -102.5, -102.25, -102., -101.75, -101.5, -101.25,
                 -101., -100.75, -100.5, -100.25, -100., -99.75, -99.5, -99.25,
                 -99., -98.75, -98.5, -98.25, -98., -97.75, -97.5, -97.25,
                 -97., -96.75, -96.5, -96.25, -96., -95.75, -95.5, -95.25,
                 -95., -94.75, -94.5, -94.25, -94., -93.75, -93.5, -93.25,
                 -93.]

    lat_grid = [37., 36.75, 36.5, 36.25, 36., 35.75, 35.5, 35.25, 35., 34.75,
                34.5, 34.25, 34., 33.75, 33.5, 33.25, 33., 32.75, 32.5, 32.25,
                32., 31.75, 31.5, 31.25, 31., 30.75, 30.5, 30.25, 30., 29.75,
                29.5, 29.25, 29., 28.75, 28.5, 28.25, 28., 27.75, 27.5, 27.25,
                27.]

    lat_to_idx = {}.fromkeys(lat_grid)
    long_to_idx = {}.fromkeys(lat_grid)

    for i, lat_ in enumerate(lat_grid):
        lat_to_idx[lat_] = i
    for j, long_ in enumerate(long_grid):
        long_to_idx[long_] = j

    print(len(lat_grid), len(long_grid))
    hail_df = dataframe[dataframe.EVENT_TYPE == "Hail"].reset_index().drop(columns=["index"])[IMPORTANT_COLS]
    hail_df["BEGIN_DATE_TIME"] = pd.to_datetime(hail_df["BEGIN_DATE_TIME"])
    hail_df["DOY"] = hail_df["BEGIN_DATE_TIME"].dt.dayofyear
    hail_df = hail_df.drop(columns=["BEGIN_DATE_TIME"])
    hail_df = hail_df[hail_df["BEGIN_LAT"] < lat[1]]
    hail_df = hail_df[hail_df["BEGIN_LAT"] > lat[0]]
    hail_df = hail_df[hail_df["BEGIN_LON"] < long[1]]
    hail_df = hail_df[hail_df["BEGIN_LON"] > long[0]]
    hail_df = hail_df.reset_index().drop(columns=["index"])
    hail_df = hail_df.apply(round_coord, axis=1)
    hail_df = hail_df.drop_duplicates().reset_index().drop(columns=["index"])
    hail_np = hail_df.to_numpy()
    grid = np.zeros((num_of_days, len(lat_grid), len(long_grid)))

    for row in hail_np:
        grid[int(row[2]), lat_to_idx[row[0]], long_to_idx[row[1]]] = 1.0
    return grid


def prepare_target_grid(path: str, lat: tuple, long: tuple, format: str = "csv"):
    target_paths = sorted(glob.glob(path + "/*." + format))
    grids = []
    if format == "csv":
        reader = pd.read_csv
    for path in target_paths:
        dataframe = reader(path)
        grids.append(get_grid(dataframe, lat, long))
    grids = np.concatenate(grids, axis=0)
    return grids


