import pandas as pd
import numpy as np

def short_date_format(row):
    row[0] = row[0][3:]
    return row

DATA_PATH = "data/ojdamage_rus.xls"
REGION = "Субъект Российской Федерации "
CUR_REGION = "Краснодарский край"
EVENTNAME_COL = "Название явления "
STARTDATE = "Дата начала "
EVENTTYPE = "Град"
MONTH_COL = "Month"
YEAR_COL = "Year"
TARGET = "target"
def prepare_data(data_path = DATA_PATH):
    data = pd.read_excel(data_path)
    hail_data = data[data[EVENTNAME_COL] == EVENTTYPE].reset_index().drop(columns = "index")[[STARTDATE, REGION]]
    hail_data = hail_data[hail_data[REGION] == CUR_REGION].reset_index().drop(columns = "index")
    hail_data[STARTDATE] = hail_data[[STARTDATE]].apply(short_date_format, axis = 1)
    hail_data = hail_data.drop_duplicates()
    hail_data[STARTDATE] = pd.to_datetime(hail_data[STARTDATE], format="%m.%Y")#, dayfirst = True)
    hail_data = hail_data.sort_values(by=[STARTDATE])
    hail_data = hail_data.drop(columns = [REGION])
    hail_data[TARGET] = np.ones(hail_data.shape[0], dtype=int)
    hail_data = hail_data.set_index([STARTDATE])
    idx = pd.date_range(min(hail_data.index), max(hail_data.index), freq='MS')
    hail_data = hail_data.reindex(idx, fill_value=0)
    return hail_data