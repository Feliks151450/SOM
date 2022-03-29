#!/usr/bin/env python
# coding: utf-8

# In[1]:


def ispan(start,end,stride = 1):
  import numpy as np
  n = int((end-start+10)/stride)
  ts = [start+stride*i for i in range(0,n)]#range(0,n)最大只能n-1
  return np.array(ts)


# In[3]:


def read_jra55(type,range,opt = True):
  # from get_path import *
  # from sub_function.get_path import*
  import sub_function.get_path as gp
  import sub_function.sub_toolbox as tb
  import xarray as xr
  
  path = gp.readpath(0)+'jra55/relv/'
  tem = []
  for i in type.year: 
    var_tem = xr.open_dataset(path+'relv.mn.'+str(i)+'.nc')
    var_tem['time'] = tb.ispan(1,12,1)
    var_tem0 = var_tem['relv'].loc[type.months,type.level,range[0]:range[1]:-1,range[2]:range[3]]
    var_tem0['year'] = i
    var_tem0.expand_dims(['year'])
    tem.append(var_tem0)
  var = xr.concat(tem, dim = 'year')
  var0 = var.rename({'time':'month'})
  return var0


# In[ ]:


def read_ERA5(type,range,opt = True):
  import sub_function.get_path as gp
  import sub_function.sub_toolbox as tb
  import xarray as xr
  path = gp.readpath(0)+'ERA5/relv/'
  tem = []
  for i in type.year: 
    var_tem = xr.open_dataset(path+'relv.mn.'+str(i)+'.nc')
    var_tem['time'] = tb.ispan(1,12,1)
    var_tem0 = var_tem['relv'].loc[type.months,type.level,range[0]:range[1]:-1,range[2]:range[3]]
    var_tem0['year'] = i
    var_tem0.expand_dims(['year'])
    tem.append(var_tem0)
  var = xr.concat(tem, dim = 'year')
  var0 = var.rename({'time':'month'})
  return var0

