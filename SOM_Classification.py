import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from SOM import train_SOM,feature_normalization,get_U_Matrix,get_winner_index,weights_PCA
from collections import defaultdict, Counter
import matplotlib.gridspec as gridspec
if __name__ == "__main__":
    
    # 读取iris数据
    datas = np.loadtxt("iris.data",delimiter=",",usecols=(0,1,2,3),dtype='float32')
    labs = np.loadtxt("iris.data",delimiter=",",usecols=(4),dtype='str')
    N,D = np.shape(datas)
    
    # 数据预处理
    datas = datas/np.linalg.norm(datas,axis=1,keepdims=True)

    # 数据切分 分为训练接和测试集
    N_train = int(np.ceil(N*0.7))
    N_test = N-N_train
    print(N_train)
    rand_index = np.random.permutation(np.arange(N))
    
    train_datas = datas[rand_index[:N_train]]
    train_labs = labs[rand_index[:N_train]]
    
    test_datas = datas[rand_index[N_train:]]
    test_labs = labs[rand_index[N_train:]]
    
    # SOM 训练
    X=7
    Y=7
    weights = train_SOM(X=X,Y=Y,N_epoch=5,datas=train_datas,sigma=0.5,init_weight_fun=weights_PCA,seed=20)
    
    # 计算输出层的每个节点上映射了哪些数据
    win_map = defaultdict(list)
    for x,lab in zip(datas,labs):
        win_map[get_winner_index(x,weights)].append(lab)
    
    win_lab = defaultdict(list)
    for key in win_map.keys():
        win_lab[key] = max(win_map[key],key=win_map[key].count)
    print(win_lab)
    
    # 进行测试：
    n_right =  0
    for i in range(N_test):
        x = test_datas[i]
        win = get_winner_index(x,weights)
        
        if win in win_lab.keys():
            det_lab = win_lab[win]
        else:
            det_lab = 'None'
        
        if det_lab == test_labs[i]:
            n_right = n_right+ 1
     
    # 计算准确率
    print('Accuracy = %.2f %%'%(n_right*100/N_test))
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # # seed 数据展示
    # columns=['area', 'perimeter', 'compactness', 'length_kernel', 'width_kernel',
                   # 'asymmetry_coefficient', 'length_kernel_groove', 'target']
    # data = pd.read_csv('seeds_dataset.txt', 
                    # names=columns, 
                   # sep='\t+', engine='python')
    # labs = data['target'].values
    # label_names = {1:'Kama', 2:'Rosa', 3:'Canadian'}
    # datas = data[data.columns[:-1]].values
    # N,D = np.shape(datas)
    # print(N,D)
    
    # # 对训练数据进行正则化处理
    # datas = feature_normalization(datas)
    
    # # SOM的训练
    # X=3
    # Y=1
    # weights = train_SOM(X=X,Y=Y,N_epoch=4,datas=datas,sigma=1.5,init_weight_fun=weights_PCA)
    
    # # 实现聚类
    
    # # 获取聚类的编号
    # index_clusters = []
    # for i in range(N):
        # x = datas[i]
        # winner = get_winner_index(x,weights)
        # index_clusters.append(winner[0]*Y+winner[1])
    
    
    # for c in np.unique(index_clusters):
        
        # ii = np.where(index_clusters==c)[0]
        
        # plt.scatter(datas[ii, 0],
                    # datas[ii, 2], label='cluster='+str(c), alpha=.7)
    # plt.legend()                
    # for i in range(X):
        # for j in range(Y):
            # plt.scatter(weights[i,j,0], weights[i,j,2], marker='x', 
                # s=80, linewidths=1, color='k')
    # plt.legend()
    # plt.show()

    
    
    
    
    
    
    
 