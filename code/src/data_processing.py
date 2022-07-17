import torch
from osgeo import gdal
import xarray as xr
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import subprocess
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import glob

def get_file_paths(path_to_data: str = 'drive/MyDrive/Belgorodskaya/*.tif', feature_names: list = ['tmax', 'tmin', 'pr']):
    """
    Filters out required features amongs terraclim dataset

    Arguments:
    path_to_data (str): path to directory that containts terraclim dataset
    feature_names (list): list of required features

    Returns:
    dict: key -- feature name; value -- list of related tif files
    """
    files_to_mosaic = glob.glob(path_to_data)
    files_to_mosaic = list(filter(lambda x: sum(fn in x for fn in feature_names) > 0, files_to_mosaic))
    file_paths = {fn: list(filter(lambda x: fn in x, files_to_mosaic)) for fn in feature_names}
    return file_paths


def get_coords_res(dataset: gdal.Dataset):
    """
    For given dataset returns position of top left corner and resolutions

    Arguments:
    dataset (osgeo.gdal.Dataset): gdal dataset

    Returns:
    dict: containts coordinates of top left corner and
       resolutions alog x and y axes
    """
    gt = dataset.GetGeoTransform()
    output = {}
    output["x"] = gt[0]
    output["y"] = gt[3]
    output["x_res"] = gt[1]
    output["y_res"] = gt[-1]
    return output


def plot_tl_positions(file_paths: list):
  """
  Viualize positions of top left corners of dataset given

  Arguments:
  file_paths (list): list of paths to files that contain datasets
  """
  tlxs = []
  tlys = []
  for fp in tqdm(file_paths):
    dataset = gdal.Open(fp, gdal.GA_ReadOnly)
    if dataset is not None:
      coords_dict = get_coords_res(dataset)
      tlxs.append(coords_dict['x'])
      tlys.append(coords_dict['y'])

  fig, ax = plt.subplots()
  fig.set_figheight(15)
  fig.set_figwidth(15)
  ax.scatter(tlxs, tlys)

  for i in range(len(tlxs)):
    ax.annotate(i, (tlxs[i], tlys[i]))
  plt.gca().set_aspect('equal', adjustable='box')
  ax.set_title("Positions of top left corners of each raster")
  ax.grid(True)


def dataset_to_np(dataset: gdal.Dataset, x_off: int, y_off: int, xsize: int, ysize: int):
  """
  Converts gdal.Dataset to numpy array
  !NB: raster bands are enumerated starting from 1!

  Arguments:
    dataset (gdal.Dataset): dataset to cast
    x_off (int): starting x position - idx
    y_off (int): starting y position - idx
    xsize (int): number of points to save in x direction
    ysize (int): number of points to save in y direction
  Returns:
    np.ndarray -- 3d tensor of information given in dataset
  """

  shape = [dataset.RasterCount, ysize, xsize]
  output = np.empty(shape)
  for r_idx in range(shape[0]):
    band = dataset.GetRasterBand(r_idx + 1)
    arr = band.ReadAsArray(x_off, y_off, xsize, ysize)
    output[r_idx, :, :] = np.array(arr)

  return output


def get_nps(feature_names, path_to_tifs, dset_num=0):
  file_paths = get_file_paths(path_to_tifs, feature_names)
  # open gdal files
  dsets = {}
  for fn in feature_names:
    dset = gdal.Open(file_paths[fn][dset_num])
    dsets[fn] = dset
  # reading into np, scaling in accordance with terraclim provided
  nps = {}
  for fn in feature_names:
    np_tmp = dataset_to_np(dsets[fn], x_off = 0, y_off = 0, xsize = dsets[fn].RasterXSize, ysize = dsets[fn].RasterYSize)
    #Scaling in accordance with dataset description
    if fn == 'tmin' or fn == 'tmax':
      nps[fn] = np_tmp * 0.1
    elif fn == 'ws':
      nps[fn] = np_tmp * 0.01
    elif fn == 'vap':
      nps[fn] = np_tmp * 0.001
    elif fn == 'seasurfacetemp':
      nps[fn] = np_tmp * 0.01
    else:
      nps[fn] = np_tmp
  
  #getting mean temp if accessible
  if 'tmin' in feature_names and 'tmax' in feature_names:
    nps['tmean'] = (nps['tmax'] + nps['tmin']) / 2
  
  return nps


TARGET_PATH = "data/ojdamage_rus.xls"
DATA_PATH = "HailProject/code/data"
REGION = "Субъект Российской Федерации "
CUR_REGION = "Москва"
EVENTNAME_COL = "Название явления "
STARTDATE = "Дата начала "
EVENTTYPE = "Град"
MONTH_COL = "Month"
YEAR_COL = "Year"
TARGET = "target"


def get_dataloader(
        feature_names: list = None,
        data_path: str = DATA_PATH,
        batch_size: int = 4,
        eco: bool = True,
        eco_len: int = 5,
        train: bool = True,
        part: int = 1
):
    r"""

    Args:
        feature_names:
        data_path:
        batch_size:
        eco:
        eco_len:
        train:
        part:

    Returns:
        dl: dataloader with data
        x: data

    Function for creating dataloader for train and test
    """
    xs = []

    hail_path = data_path + "/Hail/"
    no_hail_path = data_path + "/No Hail/"

    hail_paths = glob.glob(hail_path + "*")
    no_hail_paths = glob.glob(no_hail_path + "*")

    for p in hail_paths[(part - 1): part]:
        x = get_nps([feature_names[0]], p + "/*")
        x = x[feature_names[0]]
        x = np.nan_to_num(x)
        x = np.expand_dims(x, axis=1)
        for feature_name in feature_names[1:]:
            numpys = get_nps([feature_name], p + "/*")
            x = np.concatenate((x, np.expand_dims(numpys[feature_name], axis=1)), axis=1)
        x = torch.from_numpy(x)
        x = x.long()
        xs.append(x.unsqueeze(dim=0))

    for p in no_hail_paths[(part - 1) * 10: 10 * part]:
        x = get_nps([feature_names[0]], p + "/*")
        x = x[feature_names[0]]
        x = np.expand_dims(x, axis=1)
        for feature_name in feature_names[1:]:
            numpys = get_nps([feature_name], p + "/*")
            x = np.concatenate((x, np.expand_dims(numpys[feature_name], axis=1)), axis=1)
        x = torch.from_numpy(x)
        x = x.long()
        xs.append(x.unsqueeze(dim=0))

    x = torch.cat(xs, dim=0)
    if eco is True:
        target = [1 for _ in range(len(hail_paths[(part - 1):  part]))] + \
                 [0 for _ in range(len(no_hail_paths[(part - 1) * 10: 10 * part]))]
    else:
        target = [1 for _ in range(len(hail_paths))] + [0 for _ in range(len(no_hail_paths))]
    y = torch.tensor(target).float().reshape(-1, 1)
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size, shuffle=True)

    return dl, x


def short_date_format(row):
    row[0] = row[0][3:]
    return row


def get_target(forecasting_period: tuple,
               data_path: str = TARGET_PATH,
               region: str = CUR_REGION,
               freq: str = "Monthly"
               ):
    r"""

    Args:
        forecasting_period:
        data_path:
        region:
        freq:

    Returns:
        hail_data: pandas.DataFrame with dates with hail

    """
    data = pd.read_excel(data_path)
    hail_data = data[data[EVENTNAME_COL] == EVENTTYPE].reset_index().drop(columns="index")[[STARTDATE, REGION]]
    hail_data = hail_data[hail_data[REGION] == region].reset_index().drop(columns="index")
    if freq == "Monthly":
        hail_data[STARTDATE] = hail_data[[STARTDATE]].apply(short_date_format, axis=1)
    hail_data = hail_data.drop_duplicates()
    if freq == "Monthly":
        hail_data[STARTDATE] = pd.to_datetime(hail_data[STARTDATE], format="%m.%Y")  # , dayfirst = True)
    elif freq == "Daily":
        hail_data[STARTDATE] = pd.to_datetime(hail_data[STARTDATE], format="%d.%m.%Y")  # , dayfirst = True)
    hail_data = hail_data.sort_values(by=[STARTDATE])
    hail_data = hail_data.drop(columns=[REGION])
    hail_data[TARGET] = np.ones(hail_data.shape[0], dtype=int)
    hail_data = hail_data.set_index([STARTDATE])
    if freq == "Monthly":
        idx = pd.date_range(
            pd.to_datetime(f"01.{forecasting_period[0]}", format="%m.%Y"),
            pd.to_datetime(f"12.{forecasting_period[1]}", format="%m.%Y"),
            freq='MS')
    elif freq == "Daily":
        idx = pd.date_range(
            pd.to_datetime(f"01.01.{forecasting_period[0]}", format="%d.%m.%Y"),
            pd.to_datetime(f"31.12.{forecasting_period[1]}", format="%d.%m.%Y"),
            freq='D')
    hail_data = hail_data.reindex(idx, fill_value=0)
    return hail_data, data


def prepare_train_data(path: str, one_day: False):
    """

    Args:
        path:
        one_day:

    Returns:
        train_days

    Функция для обработки данных из файлов базы данных.
    Выдает тензор (дни, кол-во клим. перем, широта, долгота)
    """
    ds = xr.open_dataset(path, engine="cfgrib")
    vars = list(ds.data_vars)

    dropped = ["q", "z", "d2m", "cape", "sp"]
    left_vars = []

    for i, var in enumerate(vars):
        if var in dropped:
            continue
        else:
            left_vars.append(var)

    vars = left_vars

    nps = {}.fromkeys(vars)

    for var in vars:
        nps[var] = ds[var].to_numpy()

    train_days = []
    time = ds.dims["time"]
    day_var = {}.fromkeys(vars)
    del ds

    for day in range(time * 6 // 24):
        for var in vars:
            day_var[var] = []
        for hour_period in range(4):
            for var in vars:
                day_var[var].append(nps[var][day * 4 + hour_period])
        train_days.append(np.array([day_var[var] for var in vars]).reshape(-1, 41, 65))
        if one_day:
            break

    train_days = np.array(train_days)
    return train_days


def prepare_extra_feature(extra_feature_path: str):
    """

    Args:
        extra_feature_path:

    Returns:

    Аналогичная prepare_train_data, но для дополнительных подсчитанных отдельно фичей
    """
    extra_feature_paths = sorted(glob.glob(extra_feature_path + "/*.npy"))
    full_ef_ds = []
    for ef_path in extra_feature_paths:
        new_np = []
        np_ = np.load(ef_path)
        for day in range(len(np_) * 6 // 24):
            day_np = []
            for h in range(4):
                day_np.append(np_[day * 4 + h])
            new_np.append(day_np)
        full_ef_ds.append(new_np)

    full_ef_ds = np.concatenate(full_ef_ds, axis=0)
    return full_ef_ds


def prepare_runoff(runoff_path: str, one_day: bool = True):
    """

    Args:
        runoff_path:
        one_day:

    Returns:

    Аналогичная prepare_train_data, но для отдельной фичи runoff
    """
    ds = xr.open_dataset(runoff_path, engine="cfgrib")
    vars = list(ds.data_vars)
    nps = {}.fromkeys(vars)

    for var in vars:
        nps[var] = ds[var].to_numpy()[:, 1, :, :]

    train_day = []
    time = ds.dims["time"]
    day_var = {}.fromkeys(vars)
    del ds

    for day in range(time * 12 // 24):
        for var in vars:
            day_var[var] = []
        for hour_period in range(2):
            for var in vars:
                day_var[var].append(nps[var][day * 2 + hour_period])
        train_day.append(np.array([day_var[var] for var in vars]).reshape(-1, 41, 65))
        if one_day:
            break

    train_day = np.array(train_day)
    return train_day


def prepare_full_train_data(aerology_path: str,
                            land_path: str,
                            runoff_path: str,
                            extra_feature_path: str,
                            one_day: bool = False):
    """

    Args:
        aerology_path:
        land_path:
        runoff_path:
        extra_feature_path:
        one_day:

    Returns:

    Данная функция выдает полный датасет, стакая выходы функций выше.

    """
    aerology_paths = glob.glob(aerology_path + "/*.grib")
    land_paths = glob.glob(land_path + "/*.grib")
    runoff_paths = glob.glob(runoff_path + "/*.grib")
    full_train_days = []
    for a_path, l_path, ro_path in zip(sorted(aerology_paths), sorted(land_paths), sorted(runoff_paths)):
        full_train_days.append(np.concatenate([#prepare_train_data(l_path, one_day),
                                               prepare_train_data(a_path, one_day),
                                               prepare_runoff(ro_path, one_day)], axis=1))
    extra_feature_data = prepare_extra_feature(extra_feature_path)
    full_train_days = np.concatenate(full_train_days, axis=0)
    full_train_days = np.concatenate([full_train_days, extra_feature_data], axis=1)
    return full_train_days

#####################################################################################################################
#       Data preparation - fitting pipeline.                                                                        #
#                                                                                                                   #
#   1)<gribs paths> --> prepare_full_train_data() --> <tensors with dimension: (n_days, n_features, lat, long)> --> #
#   --> train_model() --> <pretrained model>;                                                                       #
#                                                                                                                   #
#   2)<cmips path> --> prepare_full_cmip_data() --> <tensors with dimension: (n_days, n_features, lat, long)>;      #
#                                                                                                                   #
#   3)prepare_target_grid() --> <grid with dimension: (n_days, lat, long)>                                          #
#                                                                                                                   #
#   4)<tensors with dimension: (n_days, n_features, long, lat)> & <pretrained model> &                              #
#     & <grid with dimension: (n_days, lat, long)> --> inference_model()                                            #
#                                                                                                                   #
#####################################################################################################################