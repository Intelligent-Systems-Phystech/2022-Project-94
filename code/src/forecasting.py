import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf,month_plot,quarter_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import SimpleExpSmoothing
import statsmodels.api as sm
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
import xgboost as xgb

# Basic imports
import datetime # manipulating date formats
import itertools
import time
import holidays
from sklearn.metrics import mean_squared_error as MSE, r2_score, mean_absolute_percentage_error as MAPE
from scipy import stats
from scipy import special

# Stats packages
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.tools import diff
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from scipy import signal

import statsmodels.api as sm

# Basic imports
from math import sqrt

# Machine learning basics
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE, r2_score
from sklearn import datasets, linear_model

def ts_to_pd(ts, transforms=[], **kwargs):
  """
  transforms 3d array of time series to dict of pd.DataFrames.
  applies specified transforms to data

  Arguments:
    ts (3darray): timeseries 0 - time, 1, 2 - spatial coordinates
    transforms (list): names for transforms: 'sp_avg', 'log'
    shift (int): shift for diff transform
    x_vic (int): vicinity over x to average over
    y_vic (int): vicinity over y to average over
    x_loc (int): location of interest -- if all time series are not required
    y_loc (int): location of interest -- if all time series are not required
  """
  y_m = lambda x: str(1958 + (x // 12)) + '-' + (str(x % 12 + 1) if x % 12 + 1 >= 10 else ('0' + str(x % 12 + 1)))
  ts_tmp = ts
  ts_tmp = ts_tmp.reshape(-1, ts_tmp.shape[-2], ts_tmp.shape[-1])
  inv_transforms = []
  if 'sp_avg' in transforms:
    try:
      ts_avged = []
      for i in range(ts.shape[0]):
        ts_avged.append(signal.convolve2d(ts[i, :, :],
        np.ones((kwargs['x_vic'], kwargs['y_vic'])),
        boundary='symm', mode='valid')
         / (kwargs['x_vic'] * kwargs['y_vic']))
      ts_tmp = np.stack(ts_avged)
    except KeyError:
      print("Spatial averaging requires `x_vic` and `y_vic` arguments.\n See docstring")

  if 'log' in transforms:
    ts_tmp = np.log(ts_tmp)
    inv_transforms.append(np.exp)
  
  
  if 'shift' in transforms:
    raise NotImplementedError("To be done........")

  if 'x_loc' in kwargs.keys() and 'y_loc' in kwargs.keys():
    i = kwargs['x_loc']
    j = kwargs['y_loc']
    ts_tmp_loc = ts_tmp[:, i, j]
    if 'boxcox' in transforms:
      ts_tmp_loc, lam = stats.boxcox(ts_tmp_loc)
      inv_transforms.append(lambda x: special.inv_boxcox(x, lam))
    df = pd.DataFrame({"Date": [y_m(i_) for i_ in range(ts.shape[0])], "val": ts_tmp_loc})
    df.set_index("Date", inplace=True)
    df.index = pd.to_datetime(df.index)

    return df, inv_transforms
  else:

    out_dfs = {}
    for i in range(ts_tmp.shape[1]):
      for j in range(ts_tmp.shape[2]):
        df = pd.DataFrame({"Date": [y_m(i) for i in range(ts.shape[0])], "val": ts[:, i, j]})
        df.set_index("Date", inplace=True)
        df.index = pd.to_datetime(df.index)
        out_dfs[(i, j)] = df

    return out_dfs, inv_transforms


def train_test_demo_split(df,train_start, train_end, test_end, demo_start):
  #train_end = '2010-01-01'
  #test_end = '2015-01-01'
  #demo_start = '2010-01-01'
  demo = df[demo_start:test_end]
  train,test = df[train_start:train_end], df[train_end:test_end]
  return train, demo, test
  #train.plot(figsize=(12,6),grid=True) 

def test_stationarity(timeseries):
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    try:
      dftest = adfuller(timeseries,maxlag=12*4, autolag='AIC')
    except ValueError:
      dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (round(dfoutput,3))

def kpss_test(series, **kw):    
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format Output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')

def sarimax(ts, all_param, exo=None):
    results = []
    for param in all_param:
        try:
            mod = SARIMAX(ts,
                          exog = exo,
                          order=param[0],
                          seasonal_order=param[1])
            res = mod.fit()
            results.append((res,res.aic,param))
            print('Tried out SARIMAX{}x{} - AIC:{}'.format(param[0], param[1], round(res.aic,2)))
        except Exception as e:
            print(e)
            continue
            
    return results

def sarimax_grid_search(p, d, q, P, D, Q, s, train, exog=None, summary=True):
  pdq = list(itertools.product(p, d, q))
  seasonal_pdq = list(itertools.product(P, D, Q, s))
  all_param = list(itertools.product(pdq,seasonal_pdq))

  all_res = sarimax(train,all_param, exog)
  all_res.sort(key=lambda x: x[1])

  res = all_res[0][0]
  if summary:
    res.plot_diagnostics(figsize=(15, 12))

    plt.show()
    print("Ljung-box p-values:\n" + str(res.test_serial_correlation(method='ljungbox')[0][1]))
    res.summary()
  return res

def plot_test_forecast(res, train, exo_train, test, exo_test, train_end, test_end, demo_start, label):
  pred_test = res.get_prediction(start=train_end,end=test_end,exog=exo_test)
  # The root mean squared error
  err = 'Mean absolute percentage error: %.2f'% MAPE(test, pred_test.predicted_mean) + \
  '\nRoot mean squared error: %.2f'% sqrt(MSE(test, pred_test.predicted_mean)) + \
  '\nR 2 score: %.2f'% r2_score(test, pred_test.predicted_mean)

  pred = res.get_prediction(start=demo_start,end=test_end,exog=exo_test)
  pred_ci = pred.conf_int()

  fig, ax = plt.subplots(figsize=(12,7))
  ax.set(ylabel=label)

  train[demo_start:].plot(ax=ax)
  test.plot(ax=ax)
  pred.predicted_mean.plot(ax=ax)
  ci = pred_ci.loc[demo_start:]
  ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='r', alpha=0.1)

  plt.figtext(0.12, -0.06, err, ha="left",fontsize=15,va='center')
  legend = ax.legend(["Train Set Observed","Test Set Observed", "Forecast", "Confidence Interval"])
  ax.grid(True)

def plot_train_test_forecast(res, train, demo_start, test, train_end, test_end):
  begin = train_end
  pred_test = res.get_prediction(start=train_end,end=test_end,exog=exo_test)

  err = 'Mean absolute percentage error: %.2f'% MAPE(test, pred_test.predicted_mean) + \
  '\nRoot mean squared error: %.2f'% sqrt(MSE(test, pred_test.predicted_mean)) + \
  '\nR 2 score: %.2f'% r2_score(test, pred_test.predicted_mean)

  pred = res.get_prediction(start=begin,end=test_end,exog=exo_test)
  pred_ci = pred.conf_int()
  #pred_test = res.get_prediction(start=train_end,end='2020-12-01')
  fig, ax = plt.subplots(figsize=(12,7))
  ax.set(ylabel='C')

  train.plot(ax=ax)
  test.plot(ax=ax)
  pred.predicted_mean.plot(ax=ax)
  ci = pred_ci.loc[demo_start:]
  ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='r', alpha=0.1)

  plt.figtext(0.12, -0.06, err, ha="left",fontsize=15,va='center')
  legend = ax.legend(["Train Set Observed","Test Set Observed", "Forecast", "Confidence Interval"])
  plt.xlim(('2000-01-01', '2020-12-01'))

def get_htc(tmean, pr, transforms = [[np.exp], [np.exp]]):
  tmean_inv = tmean
  pr_inv = pr
  for t_i in range(len(transforms[0])):
    tmean_inv = transforms[len(transforms) - 1 - t_i](tmean_inv)
  for t_i in range(len(transforms[1])):    
    pr_inv = transforms[len(transforms) - 1 - t_i](pr_inv)
  mask = (tmean_inv > 10).values
  pr_masked = (pr_inv * mask).groupby(by = [pr_inv.index.year]).sum()
  tmean_masked = (tmean_inv * mask).groupby(by = [tmean_inv.index.year]).sum()
  htc = 10 * pr_masked / (30 * tmean_masked)

  return htc

def pipeline(tmean, pr, ws, vap, htc, x_loc, y_loc, x_vic, y_vic, train_end, test_end, demo_start):

  # preparing datasets
  df_tmean, inv_tf_tmean = ts_to_pd(tmean, ['sp_avg'], x_vic=10, y_vic=10, x_loc=x_loc, y_loc=y_loc)
  df_pr, inv_tf_pf = ts_to_pd(pr, ['boxcox', 'sp_avg'], x_vic=10, y_vic=10, x_loc=x_loc, y_loc=y_loc)
  df_ws, inv_tf_ws = ts_to_pd(ws, ['boxcox', 'sp_avg'], x_vic=10, y_vic=10, x_loc=x_loc, y_loc=y_loc)
  df_vap, inv_tf_vap = ts_to_pd(vap, ['boxcox', 'sp_avg'], x_vic=10, y_vic=10, x_loc=x_loc, y_loc=y_loc)

  # preparing htc dataset separately, since its yearly 
  htc_avged = []
  for i in range(htc.shape[0]):
    htc_avged.append(signal.convolve2d(htc[i, : ,:], np.ones((10, 10)), boundary='symm', mode='valid') / 100)
  htc_avged = np.stack(htc_avged)

  df_htc = pd.DataFrame({'val': htc_avged[:, x_loc, y_loc],  'Date': [str(1958 + i) for i in range(htc_avged.shape[0])]})
  df_htc = df_htc.set_index("Date")
  df_htc.index = pd.to_datetime(df_htc.index)

  #train-test splitting
  train_start = '1958-01-01'
  #train_end = '2010-01-01'
  #test_end = '2020-01-01'
  #demo_start = '2010-01-01'
  # df_pr_log_diff_6 = (np.log(df_pr) - np.log(df_pr).shift(6)).dropna()#.val
  # df_pr_log_diff_6 = np.log(df_pr)
  train_pr, demo_pr, test_pr = train_test_demo_split(df_pr, train_start, train_end, test_end, demo_start)
  train_ws, demo_ws, test_ws = train_test_demo_split(df_ws, train_start, train_end, test_end, demo_start)
  train_vap, demo_vap, test_vap = train_test_demo_split(df_vap, train_start, train_end, test_end, demo_start)
  train_tmean, demo_tmean, test_tmean = train_test_demo_split(df_tmean, train_start, train_end, test_end, demo_start)
  train_htc, demo_htc, test_htc = train_test_demo_split(df_htc, train_start, train_end, test_end, demo_start)

  # plain htc
  res_htc = ARIMA(train_htc, order=(19, 2, 0)).fit()
  pred_test_htc = res_htc.get_prediction(start=train_end,end=test_end,exog=None)
  errs_p = [MAPE(test_htc, pred_test_htc.predicted_mean),\
        sqrt(MSE(test_htc, pred_test_htc.predicted_mean)),\
        r2_score(test_htc, pred_test_htc.predicted_mean)]

  pred_htc = res_htc.get_prediction(start=demo_start,end=test_end,exog=None)
  pred_ci_htc = pred_htc.conf_int()
  ci_htc = pred_ci_htc.loc[demo_start:]
  #cis_list.append([ci_htc.iloc[:,0], ci_htc.iloc[:,1]])
  pred_htc_p = pred_htc.predicted_mean

  # sarimax
  p,d,q = [1],[1],[2] 
  P,D,Q,s = [1],[1],[2],[12] #season 2 years, small p d q 1 1 2 24
  # list of all parameter combos
  res_tmean = sarimax_grid_search(p, d, q, P, D, Q, s, train_tmean, exog=None, summary=False)

  p,d,q = [2],[0],[0]
  P,D,Q,s = [5],[0],[0],[12]
  # list of all parameter combos
  res_ws = sarimax_grid_search(p, d, q, P, D, Q, s, train_ws, exog=None, summary=False)

  train_tmean_ws = train_tmean.copy()
  train_tmean_ws['val1'] = train_ws.val

  p,d,q = [2],[0],[0]
  P,D,Q,s = [5],[0],[0],[12]
  # 3 1 1 x 3 1 0 x 60
  res_pr = sarimax_grid_search(p, d, q, P, D, Q, s, train_pr, exog=train_tmean_ws, summary=False)

  exog_pred = pd.DataFrame(res_tmean.get_prediction(demo_start, test_end).predicted_mean[:-1])
  exog_pred['val1'] = res_ws.get_prediction(demo_start, test_end).predicted_mean[:-1]

  mask_test = ((test_tmean > 10).values)#.flatten()
  test_pr_masked = inv_tf_pf[0](test_pr) * mask_test
  test_tmean_masked = (test_tmean * mask_test)
  true_htc = 10 * test_pr_masked.groupby(by = [test_pr_masked.index.year]).sum() / (30 * test_tmean_masked.groupby(by = [test_tmean_masked.index.year]).sum())

  test_tmean_pred = res_tmean.get_prediction(train_end, test_end).predicted_mean
  test_pr_pred = res_pr.get_prediction(train_end, test_end, exog=exog_pred).predicted_mean
  mask_pred = (test_tmean_pred > 10).values
  #pred_pr_masked = (np.exp(test_pr_pred)) * mask_pred
  pred_pr_masked = (inv_tf_pf[0](test_pr_pred)) * mask_pred

  pred_pr_masked *= mask_pred
  pred_tmean_masked = (test_tmean_pred * mask_pred)
  pred_htc = 10 * pred_pr_masked.groupby(by = [pred_pr_masked.index.year]).sum() / (30 * pred_tmean_masked.groupby(by = [pred_tmean_masked.index.year]).sum())

  errs_s = [MAPE(true_htc[:-1], pred_htc[:-1]),\
        np.sqrt(MSE(true_htc[:-1], pred_htc[:-1])),\
        r2_score(true_htc[:-1], pred_htc[:-1])]
  pred_htc_s = pred_htc[:-1]

  # xgboost
  X, y = construct_shifted_data(df_tmean, df_pr, df_ws, df_vap, df_htc, tshift=12, pred_shift=1, standartize=False)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

  model = xgb.XGBRegressor()
  model.fit(X_train, y_train)

  pred_htc = model.predict(X_test)
  errs_x = [MAPE(y_test, pred_htc), np.sqrt(MSE(y_test, pred_htc)), r2_score(y_test, pred_htc)]
  pred_htc_x = pred_htc

  return (errs_p, errs_s, errs_x), (pred_htc_p, pred_htc_s, pred_htc_x)

def construct_shifted_data(df_tmean, df_pr, df_ws, df_vap, df_htc, tshift=12, pred_shift=1, standartize=False):
  X = []
  y = []
  assert len(df_tmean) == len(df_pr) and len(df_pr) == len(df_ws) and len(df_ws) == len(df_vap)
  for i in list(range(len(df_tmean)))[::12][tshift // 12:-pred_shift]:
    obj = np.hstack([df_tmp.iloc[i - tshift: i].values.flatten() for df_tmp in [df_tmean, df_pr, df_ws, df_vap]])
    X.append(obj)
    y.append(df_htc.iloc[i // 12 + pred_shift].values)
  X = np.stack(X)
  y  = np.stack(y).flatten()
  if standartize:
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y.reshape(-1, 1)).flatten()
    return X, y, sc_X, sc_y
  else:
    return X, y


### MIGHT BE LEGACY
def inverse_logdiff_l(df_log_diff_l, df_first_l, l = 6):
  """
  Performs inverse transform of log(x) - log(x).shift(l)
  
  Arguments:
    df_log_diff_l (pd.DataFrame): array after logg diff l
    df_first_l (pd.DataFrame): array of first used for
       computing first l elements of transofrms
    l (int): shift length

  """
  df_inv = pd.concat([df_first_l, df_log_diff_l.copy()])
  for i in range(l, len(df_inv)):
    n = i // l
    k = i % l
    df_inv.iloc[i] = np.exp(sum([df_log_diff_l.iloc[l * n + k - l * j] for j in range(1, n+1)])) * df_first_l.iloc[k]
  return df_inv