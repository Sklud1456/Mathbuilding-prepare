# 导入可能要用到的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # 可视化图形调用库
from tqdm import *
import heapq


def ShowGRAHeatMap(data):
    # 色彩集
    colormap = plt.cm.RdBu
    plt.figure(figsize=(18,16))
    plt.title('Person Correlation of Features',y=1.05,size=18)
    sns.heatmap(data.astype(float),linewidths=0.1,vmax=1.0,square=True,\
               cmap=colormap,linecolor='white',annot=True)
    plt.show()

# 无量纲化
def dimensionlessProcessing(df_values, df_columns):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    res = scaler.fit_transform(df_values)
    return pd.DataFrame(res, columns=df_columns)


# 求第一列(影响因素)和其它所有列(影响因素)的灰色关联值
def GRA_ONE(data, m=0):  # m为参考列
    # 标准化
    # data = dimensionlessProcessing(data.values, data.columns)
    # 参考数列
    std = data.iloc[:, m]
    # 比较数列
    ce = data.copy()

    n = ce.shape[0]
    m = ce.shape[1]

    # 与参考数列比较，相减
    grap = np.zeros([n, m])
    for i in range(m):
        for j in range(n):
            grap[j, i] = abs(ce.iloc[j, i] - std[j])

    # 取出矩阵中的最大值和最小值
    mmax = np.amax(grap)
    mmin = np.amin(grap)
    ρ = 0.5  # 灰色分辨系数

    # 计算值
    grap = pd.DataFrame(grap).applymap(lambda x: (mmin + ρ * mmax) / (x + ρ * mmax))

    # 求均值，得到灰色关联值
    RT = grap.mean(axis=0)
    return pd.Series(RT)


# 调用GRA_ONE，求得所有因素之间的灰色关联值
def GRA(data):
    data = dimensionlessProcessing(data.values, data.columns)
    list_columns = np.arange(data.shape[1])
    df_local = pd.DataFrame(columns=list_columns)
    # for i in tqdm(np.arange(data.shape[1])):
    df_local.iloc[:, 729] = GRA_ONE(data, m=729)
    # for i in tqdm(np.arange(data.shape[1])):
    #     df_local.iloc[:, i] = GRA_ONE(data, m=i)
    return df_local

# 读取数据
yao1 = pd.read_excel("Molecular_Descriptor.xlsx",index_col=0)
yao2 = pd.read_excel("ERα_activity.xlsx",index_col=0)
data1=pd.merge(yao1,yao2,on='SMILES')
# print(yao2.iloc[:,1])
del data1['pIC50']
key=data1.keys()

#0.002~1区间归一化
[m,n]=data1.shape #得到行数和列数
data2=data1.astype('float').values
data3=data2
print(data2)

ymin=0.002
ymax=0.098
for j in range(0,n):
    d_max=max(data2[:,j])
    d_min=min(data2[:,j])
    temp=(ymax-ymin)*(data2[:,j]-d_min)/(d_max-d_min)+ymin
    if True in np.isnan(temp):
        # print("you nan!!")
        continue
    else:
        data3[:,j]=temp

print(data3.shape)
print(data3[:,729])
# 得到其他列和参考列相等的绝对值
for i in tqdm(range(0,729)):
    data3[:,i]=np.abs(data3[:,i]-data3[:,729])
    # print(i,"  ",data3[:,i])

#得到绝对值矩阵的全局最大值和最小值
data4=np.array(data3[:,0:729])
d_max=np.max(data4)
d_min=np.min(data4)

print(data4)
print("max",d_max)
print("min",d_min)


a=0.5
data4=(d_min+a*d_max)/(data4+a*d_max)
xishu=np.mean(data4, axis=0)
index=heapq.nlargest(20, range(len(xishu)), xishu.take)
print(index)
for i in index:
    print(key[i],end=",")
    print(xishu[i])


index=heapq.nlargest(200, range(len(xishu)), xishu.take)
newkey=[]
for i in index:
    newkey.append(key[i])
print(newkey)
# data5=data1[newkey]
# data5=pd.merge(data5,yao2,on='SMILES')
# # print(yao2.iloc[:,1])
# del data5['IC50_nM']
# print(data5)
# data5.to_csv("aftergrey.csv")


# ShowGRAHeatMap(data_gra)