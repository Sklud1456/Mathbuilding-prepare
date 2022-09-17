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
        'eta': 0.05,
        'verbosity': 1,
        'random_state': 42,
        'objective': 'reg:linear',
        'tree_method': 'gpu_hist'
    }

    def myFeval(preds, dtrain):
        labels = dtrain.get_label()
        return 'error', math.sqrt(mean_squared_log_error(preds, labels))
    print("XGBoost 训练 & 预测")
    xgb_train = xgb.DMatrix(X_train, y_train)
    model = xgb.train(param, xgb_train, num_boost_round=300, feval=myFeval)
    predict = model.predict(xgb.DMatrix(X_test))
    return predict

key=['ATSc2','ATSc3','ATSc5','BCUTc-1l','BCUTc-1h','VC-5','SdssC','minHBa','minHBint5','minHBint10','minsssN','maxssO','MAXDP','ETA_Shape_Y','MDEC-23','MDEC-33','MLFER_A','WTPT-4','XLogP']


data=pd.read_csv("aftergrey.csv",index_col=0)
feature=data[key]   #自变量
target=data['pIC50']    #因变量
x_train, x_test, y_train, y_test = train_test_split(feature, target, random_state=4)

y_predict=xgboost(x_train,y_train,x_test)
print("mse:",mean_squared_error(y_test,y_predict))

testdata=pd.read_excel("testdata.xlsx",index_col=0)
testdata=testdata[key]
test_predict=xgboost(x_train,y_train,testdata)
print("test值",test_predict)