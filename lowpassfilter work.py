# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:20:18 2022

@author: DSH
"""
import pycwt
import math
from scipy import interpolate
import numpy as np 
import xlrd
import xarray as xr
import sys
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm
from scipy import stats
import scipy
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

plt.style.use('ggplot')
#载入EXCEL数据
file = xlrd.open_workbook('F:/lowpass filter/for_lowpass_filter.xlsx')
sheet = file.sheets()[0]
dt_sic = np.array((sheet.col_values(7))[1:42],dtype = np.float64)
dt_oht_bso_aso = np.array((sheet.col_values(8))[1:42], dtype = np.float64)
dt_sic_z300 = np.array((sheet.col_values(12))[1:41], dtype = np.float64)
dt_z300 = np.array((sheet.col_values(13))[1:41],dtype = np.float64)

#载入butterworth滤波器
sys.path.append('F:/lowpass filter/lowpassfilter.py') #
import lowpassfilter

#分别对海洋部分3-30年时间窗口进行低通滤波；
dt_sic_low = np.empty(shape=(41,28))
for i in range(3,31):   
    dt_sic_low[:,i-3] = lowpassfilter.butter_lowpass_filter(dt_sic, 1/i, 1)
dt_sic_low_2 = lowpassfilter.butter_lowpass_filter(dt_sic, 1/2.0001, 1)
dt_sic_low = np.concatenate((dt_sic_low_2.reshape(41,1),dt_sic_low),axis = 1)

dt_oht_bso_aso_low = np.empty(shape=(41,28))
for j in range(3,31):   
    dt_oht_bso_aso_low[:,j-3] = lowpassfilter.butter_lowpass_filter(dt_oht_bso_aso, 1/j, 1)
dt_oht_bso_aso_low_2 = lowpassfilter.butter_lowpass_filter(dt_oht_bso_aso, 1/2.0001, 1)    
dt_oht_bso_aso_low = np.concatenate((dt_oht_bso_aso_low_2.reshape(41,1),dt_oht_bso_aso_low),axis = 1)

def corr(x,y):
    r_value = pearsonr(x,y)
    r = r_value[0]
    p = r_value[1]
    return r
#计算海洋部分不同滤波时间窗口对应的相关性
r_dt_sic_low_dt_oht_bso_aso_low = np.empty(shape = (29))
for k in range(29):
    r_dt_sic_low_dt_oht_bso_aso_low[k] = corr(dt_sic_low[:,k],dt_oht_bso_aso_low[:,k])
    
#定义有效自由度函数 x y 分别是载入变量，neff代表有效自由度，t代表t检验值，p代表p值
def effective_freedom(x,y):
    r_value = pearsonr(x,y)
    r = r_value[0]
    acf_value_x = sm.tsa.acf(x, nlags = 1)
    acf_value_x = acf_value_x[1:]
    acf_value_y = sm.tsa.acf(y, nlags = 1)
    acf_value_y = acf_value_y[1:]
    neff = len(x) * (1 - acf_value_x * acf_value_y)/(1 + acf_value_x * acf_value_y )
    t = (abs(r) * math.sqrt(neff -2 ))/math.sqrt(1 - r**2)
    p = stats.t.sf(t,neff)
    return neff,t,p

#计算海洋部分每组滤波序列相关性的的有效自由度
neff_haiyang = np.empty(shape = (29))
t_haiyang = np.empty(shape = (29))
p_haiyang = np.empty(shape = (29))
for n in range(29):
    neff_haiyang[n],t_haiyang[n],p_haiyang[n] = effective_freedom(dt_sic_low[:,n], dt_oht_bso_aso_low[:,n])

#分别对大气部分3-30年时间窗口进行低通滤波；
dt_sic_z300_low = np.empty(shape=(40,28))
for i in range(3,31):   
    dt_sic_z300_low[:,i-3] = lowpassfilter.butter_lowpass_filter(dt_sic_z300, 1/i, 1)
dt_sic_z300_low_2 = lowpassfilter.butter_lowpass_filter(dt_sic_z300, 1/2.0001, 1)
dt_sic_z300_low = np.concatenate((dt_sic_z300_low_2.reshape(40,1),dt_sic_z300_low),axis = 1)

dt_z300_low = np.empty(shape=(40,28))
for j in range(3,31):   
    dt_z300_low[:,j-3] = lowpassfilter.butter_lowpass_filter(dt_z300, 1/j, 1)
dt_z300_low_2 = lowpassfilter.butter_lowpass_filter(dt_z300, 1/2.0001, 1)
dt_z300_low = np.concatenate((dt_z300_low_2.reshape(40,1),dt_z300_low),axis = 1)

#计算大气部分不同滤波时间窗口对应的相关性   
r_dt_sic_z300_low_dt_z300_low = np.empty(shape = (29))
for k in range(29):
    r_dt_sic_z300_low_dt_z300_low[k] = corr(dt_sic_z300_low[:,k],dt_z300_low[:,k])

#计算大气部分每组滤波序列相关性的的有效自由度 
neff_daqi = np.empty(shape = (29))
t_daqi = np.empty(shape = (29))
p_daqi = np.empty(shape = (29))
for n in range(29):
    neff_daqi[n],t_daqi[n],p_daqi[n] = effective_freedom(dt_sic_z300_low[:,n], dt_z300_low[:,n]) 
    