import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from SOM import train_SOM,feature_normalization,get_U_Matrix,get_winner_index,weights_PCA
from collections import defaultdict, Counter
import matplotlib.gridspec as gridspec
if __name__ == "__main__":
    
    # seed 数据展示
    columns=['area', 'perimeter', 'compactness', 'length_kernel', 'width_kernel',
                   'asymmetry_coefficient', 'length_kernel_groove', 'target']
    data = pd.read_csv('seeds_dataset.txt', 
                    names=columns, 
                   sep='\t+', engine='python')
    labs = data['target'].values
    label_names = {1:'Kama', 2:'Rosa', 3:'Canadian'}
    datas = data[data.columns[:-1]].values
    N,D = np.shape(datas)
    print(N,D)
    
    # 对训练数据进行正则化处理
    datas = feature_normalization(datas)
    
    # SOM的训练
    X=9
    Y=9
    weights = train_SOM(X=X,Y=Y,N_epoch=4,datas=datas,sigma=1.5,init_weight_fun=weights_PCA)
    
    # 获取UMAP
    UM = get_U_Matrix(weights)
    
    print(UM)
    # '''画散点图'''
    
    # 显示UMAP
    plt.figure(1,figsize=(9, 9))
    plt.pcolor(UM.T, cmap='bone_r')  # plotting the distance map as background
    plt.colorbar()
    
    markers = ['o', 's', 'D']
    colors = ['C0', 'C1', 'C2']
    
    # 计算每个样本点投射后的坐标
    w_x, w_y = zip(*[get_winner_index(d,weights) for d in datas])
    w_x = np.array(w_x)
    w_y = np.array(w_y)
   
    # 分别把每一类的散点在响应的方格内进行打印（+随机位置偏移）
    for c in np.unique(labs):
        idx_target = (labs==c)
        plt.scatter(w_x[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8,
                    w_y[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8, 
                    s=50, c=colors[c-1], label=label_names[c])
    plt.legend(loc='upper right')
    plt.grid()
    # plt.show()
    
    
    ''' 画饼图'''
    # 计算输出层的每个节点上映射了哪些数据
    win_map = defaultdict(list)
    for x,lab in zip(datas,labs):
        win_map[get_winner_index(x,weights)].append(lab)
    
    # 统计每个输出节点上，映射了各类数据、各多少个
    for pos in win_map: 
        win_map[pos] = Counter(win_map[pos])
    
    fig = plt.figure(2,figsize=(9, 9))
    # 按照 X,Y对画面进行分格
    the_grid = gridspec.GridSpec(Y, X, fig)
    print(the_grid)
    
    # 在每个格子里面画饼图
    for pos in win_map.keys():
        label_fracs = [win_map[pos][l] for l in label_names.keys()]

        plt.subplot(the_grid[Y-1-pos[1],
                             pos[0]], aspect=1)
        patches, texts = plt.pie(label_fracs)
        

    plt.legend(labels = label_names.values(), loc='upper left',bbox_to_anchor=(-6, 10))
    # plt.savefig('resulting_images/som_seed_pies.png')
    plt.show()
    
    
    
    
    
    
 