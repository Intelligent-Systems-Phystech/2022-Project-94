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

#####################
#                   #
# Дописать get_grid #
#                   #
#####################
def get_grid(dataframe: pd.DataFrame, state: str):
    hail_df = dataframe[dataframe.STATE == state]
    hail_df = hail_df
    grid = np.zeros()
    pass


def prepare_target_grid(path: str, format: str, state: str):
    target_paths = glob.glob(path + "/*." + format)
    grids = []
    if format == "csv":
        reader = pd.read_csv
    for path in target_paths:
        dataframe = reader(path)
        grids.append(get_grid(dataframe, state))
    grids = np.concatenate(grids, axis=0)
    return grids


