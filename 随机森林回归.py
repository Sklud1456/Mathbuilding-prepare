import pandas as pd
from tqdm import *
import datetime

from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import  LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import joblib

def start():
    print("训练开始")
    time = datetime.datetime.now()
    print(time)

def end():
    print("训练结束")
    time = datetime.datetime.now()
    print(time)


key=['ATSc2','ATSc3','ATSc5','BCUTc-1l','BCUTc-1h','VC-5','SdssC','minHBa','minHBint5','minHBint10','minsssN','maxssO','MAXDP','ETA_Shape_Y','MDEC-23','MDEC-33','MLFER_A','WTPT-4','XLogP']


data=pd.read_csv("aftergrey.csv",index_col=0)


feature=data[key]   #自变量
target=data['pIC50']    #因变量
# start()
#
#
# # 1 获取数据集
# x_train, x_test, y_train, y_test = train_test_split(feature, target, random_state=4)
#
# # 2 预估器实例化
# estimator = RandomForestRegressor(n_estimators=200, random_state=0)
#
# # 3 指定超参数集合
# param_dict = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
#
# # 4 加入超参数网格搜索和交叉验证
# estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)
#
# # 5 训练数据集
# estimator.fit(x_train, y_train)
#
# # 6 模型评估
# y_predict = estimator.predict(x_test)
# print("y_predict:\n", y_predict)
# print("直接对比真实值和预测值：\n", y_test == y_predict)
#
# score = estimator.score(x_test, y_test)
# print("准确率为：\n", score)
#
# joblib.dump(estimator, "train_model.m")
# # clf = joblib.load("train_model.m")
#
# end()

rfr = RandomForestRegressor(n_estimators=200, random_state=0)
rfr=joblib.load("train_model.m")

x_train, x_test, y_train, y_test = train_test_split(feature, target, random_state=4)
y_predict = rfr.predict(x_test)
print("mse:",mean_squared_error(y_test,y_predict))
# testdata=pd.read_excel("testdata.xlsx",index_col=0)
# testdata=testdata[key]
# print(testdata)
# test_predict=rfr.predict(testdata)
# print("test值",test_predict)