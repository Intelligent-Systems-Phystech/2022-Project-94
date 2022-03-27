import torch
from osgeo import gdal

import matplotlib.pyplot as plt
import os, glob
from tqdm import tqdm
import subprocess
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


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
REGION = "Субъект Российской Федерации "
CUR_REGION = "Тамбовская область"
EVENTNAME_COL = "Название явления "
STARTDATE = "Дата начала "
EVENTTYPE = "Град"
MONTH_COL = "Month"
YEAR_COL = "Year"
TARGET = "target"


def get_traindl(
        forecasting_period: tuple,
        feature_name: str,
        data_path: str,
        target_path: str = TARGET_PATH):

    x = get_nps([feature_name], data_path + f"/{forecasting_period[0]}/*.tif")
    x = x[feature_name]
    for year in range(forecasting_period[0] + 1, forecasting_period[1] + 1):
        numpys = get_nps([feature_name], data_path + f"/{year}/*.tif")
        x = np.concatenate((x, numpys[feature_name]))

    x = torch.from_numpy(x)
    target = get_target(forecasting_period, target_path)
    y = target.to_numpy()
    y = torch.from_numpy(y).float()
    x = x.long()
    train_ds = TensorDataset(x, y)

    batch_size = 4
    train_dl = DataLoader(train_ds, batch_size)

    return train_dl


def get_testdl(
        forecasting_period: tuple,
        feature_name: str,
        data_path: str,
        target_path: str = TARGET_PATH):

    x = get_nps([feature_name], data_path + f"/{forecasting_period[0]}/*.tif")
    x = x[feature_name]
    for year in range(forecasting_period[0] + 1, forecasting_period[1] + 1):
        numpys = get_nps([feature_name], data_path + f"/{year}/*.tif")
        x = np.concatenate((x, numpys[feature_name]))

    x = torch.from_numpy(x)
    target = get_target(forecasting_period, target_path)
    y = target.to_numpy()
    y = torch.from_numpy(y).float()
    x = x.long()
    test_ds = TensorDataset(x, y)

    batch_size = 4
    test_dl = DataLoader(test_ds, batch_size)

    return test_dl


def short_date_format(row):
    row[0] = row[0][3:]
    return row


def get_target(forecasting_period: tuple, data_path: str = TARGET_PATH, region: str = CUR_REGION):
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
    idx = pd.date_range(
        pd.to_datetime(f"01.{forecasting_period[0]}", format="%m.%Y"),
        pd.to_datetime(f"12.{forecasting_period[1]}", format="%m.%Y"),
        freq='MS')
    hail_data = hail_data.reindex(idx, fill_value=0)
    return hail_data
