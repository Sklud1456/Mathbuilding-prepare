import pandas as pd
from tqdm import *
import datetime

from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import  LinearRegression
from sklearn.ensemble import RandomForestRegressor

def start():
    print("训练开始")
    time = datetime.datetime.now()
    print(time)

def end():
    print("训练结束")
    time = datetime.datetime.now()
    print(time)


class RandomForestRegressorWithCoef(RandomForestRegressor):
    def fit(self,*args,**kwargs):
        super(RandomForestRegressorWithCoef,self).fit(*args,**kwargs)
        self.coef_=self.feature_importances_

data=pd.read_csv("aftergrey.csv",index_col=0)

feature=data.iloc[:,:200]   #自变量
key=feature.keys()
target=data['pIC50']    #因变量

for i in range(1):
    print("hajimei:",i)
    rf=RandomForestRegressorWithCoef(n_estimators=500,min_samples_leaf=5,n_jobs=-1,random_state=42)
    rfecv=RFECV(estimator=rf,step=1,cv=2)
    start()
    selector=rfecv.fit(feature,target)
    print("RFECV选择出的特征个数",rfecv.n_features_)
    feature_rank=rfecv.ranking_
    print("特征优先级",feature)
    print("gridscore得分",rfecv.grid_scores_)
    print("scoring得分",rfecv.scoring)
    index=[]
    for i in range(len(feature_rank)):
        if feature_rank[i]<=20:
            index.append((i,feature_rank[i]))
    s_key = []
    for i in index:
        s_key.append([key[i[0]],i[1]])
    print(s_key)
    for i in range(21):
        for j in s_key:
            if j[1]==i:
                print(j)
    end()

    rf1 = RandomForestRegressorWithCoef(n_estimators=500, min_samples_leaf=5, n_jobs=-1,random_state=42)
    rfe=RFE(estimator=rf1,n_features_to_select=30)
    start()
    select_feature=rfe.fit(feature,target)

    result=rfe.get_support()
    index=[]

    for i in range(len(result)):
        if result[i]==True:
            index.append(i)

    print(index)
    s_key=[]
    for i in index:
        s_key.append(key[i])
    print(s_key)
    end()













