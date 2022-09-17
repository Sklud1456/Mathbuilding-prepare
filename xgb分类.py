import pandas as pd
import numpy as np
import random
import gc
import time
import logging
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
import math
from sklearn.model_selection import KFold

def xgboost(X_train, y_train, X_test):
    # prefix = "xgb_"
    param = {
        'max_depth': 10,
        'eta': 0.01,
        'verbosity': 1,
        'random_state': 42,
        'objective': 'binary:logistic',
        'tree_method': 'auto'
    }

    def myFeval(preds, dtrain):
        labels = dtrain.get_label()
        return 'error', math.sqrt(mean_squared_log_error(preds, labels))
    print("XGBoost 训练 & 预测")
    xgb_train = xgb.DMatrix(X_train, y_train)
    model = xgb.train(param, xgb_train, num_boost_round=300, feval=myFeval)
    predict = model.predict(xgb.DMatrix(X_test))
    return predict

key=['ATSc2','BCUTc-1h','SCH-7','SsOH','minHBa','mindssC','maxsCH3','ETA_dEpsilon_C','ETA_BetaP','ETA_BetaP_s','ETA_Eta_F_L','ETA_EtaP_B_RC','FMF','MLFER_BH','WTPT-4','WTPT-5']


data1=pd.read_excel("Molecular_Descriptor.xlsx",index_col=0)
feature=data1[key]   #自变量
data2=pd.read_excel("ADMET.xlsx",index_col=0)

target=data2['MN']    #因变量
x_train, x_test, y_train, y_test = train_test_split(feature, target, random_state=4)

y_predict=xgboost(x_train,y_train,x_test)
print("mse:",mean_squared_error(y_test,y_predict))
pridect1=[]
for i in y_predict:
    if i>=0.5:
        pridect1.append(1)
    else:
        pridect1.append(0)

cnt=0
for i in range(len(pridect1)):
    if pridect1[i]==y_test[i]:
        cnt+=1


print("准确率：",cnt/len(pridect1))


testdata=pd.read_excel("testdata.xlsx",index_col=0)
testdata=testdata[key]
test_predict=xgboost(x_train,y_train,testdata)
print("test值",test_predict)