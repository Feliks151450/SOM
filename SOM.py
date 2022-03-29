import  numpy as np
import xarray as xr
import random
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
# import sub_function.sub_toolbox as tb
from sub_function.sub_toolbox import*
import SOM_sub as SOM
import importlib
    

type = xr.DataArray(5)
type.attrs['year'] = ispan(1979,2018,1)
type.attrs['level'] = 200
type.attrs['months'] = [7,8]
ranges = [0,60,100,160]
# import sub_function.sub_toolbox as tb
# var = tb.read_ERA5(type,ranges,True)
# from sub_function.sub_toolbox import *
var = read_jra55(type,ranges,True)
# var
# var.loc[1998,:,10,110]
var_JA = var.sum(dim = 'month')
var_JA_reshape = var_JA.stack(spatial = ('lat','lon'))
var_JA_reshape
N,D = np.shape(var_JA_reshape) #查看数组大小
datas = var_JA_reshape.values
# 对训练数据进行正则化处理
datas = SOM.feature_normalization(datas)

# SOM的训练
weights = SOM.train(X=9,Y=9,N_epoch=4,datas=datas,sigma=1.5,init_weight_fun=weights_PCA)

# 获取UMAP
UM = SOM.get_U_Matrix(weights)

plt.figure(figsize=(9, 9))
plt.pcolor(UM.T, cmap='bone_r')  # plotting the distance map as background
plt.colorbar()

markers = ['o', 's', 'D']
colors = ['C0', 'C1', 'C2']

for i in range(N):
    x = datas[i]
    w = SOM.get_winner_index(x,weights)
    i_lab = 1#labs[i]-1
    
    plt.plot(w[0]+.5, w[1]+.5, markers[i_lab], markerfacecolor='None',
         markeredgecolor=colors[i_lab], markersize=12, markeredgewidth=2)

plt.show()