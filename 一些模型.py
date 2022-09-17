import pandas as pd
import numpy as np

from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
import joblib

def change(predict):
    result=[]
    for i in predict:
        if(i[0]>i[1]):
            result.append(0)
        else:
            result.append(1)
    return result


key=['ATSc2','BCUTc-1h','SCH-7','SsOH','minHBa','mindssC','maxsCH3','ETA_dEpsilon_C','ETA_BetaP','ETA_BetaP_s','ETA_Eta_F_L','ETA_EtaP_B_RC','FMF','MLFER_BH','WTPT-4','WTPT-5']
data1=pd.read_excel("Molecular_Descriptor.xlsx",index_col=0)
feature=data1[key]   #自变量
data2=pd.read_excel("ADMET.xlsx",index_col=0)

target=data2['MN']    #因变量
x_train, x_valid, y_train, y_valid = train_test_split(feature, target, random_state=4)


#朴素贝叶斯
BNB=BernoulliNB()
BNB.fit(x_train,y_train)
score1=BNB.score(x_valid,y_valid)
predicted1=np.array(BNB.predict_proba(x_valid))
print("朴素贝叶斯log损失为 %f" %(log_loss(y_valid,predicted1)))
print(y_valid)
predicted1=change(predicted1)
print(predicted1)
print("朴素贝叶斯mse为 %f" %(mean_squared_error(y_valid,predicted1)))
print("朴素贝叶斯的准确率为：",score1)

#逻辑回归
LR = LogisticRegression(C=0.1)
LR.fit(x_train, y_train)    #根据给定的训练数据拟合模型
score2 = LR.score(x_valid, y_valid)
predicted2 = np.array(LR.predict_proba(x_valid))
print("逻辑回归log损失为 %f" %(log_loss(y_valid, predicted2)))
print("逻辑回归mse为 %f" %(mean_squared_error(y_valid,predicted1)))
print('逻辑回归准确率：', score2)

# 决策树
DT= DecisionTreeClassifier()
DT.fit(x_train, y_train)    #根据给定的训练数据拟合模型
score3 = DT.score(x_valid, y_valid)
predicted3 = np.array(DT.predict_proba(x_valid))
print("决策树log损失为 %f" %(log_loss(y_valid, predicted3)))
print("决策树mse为 %f" %(mean_squared_error(y_valid,predicted1)))
print('决策树准确率：', score3)

#支持向量机SVM
modelSVM = SVC(kernel='linear', C=100,probability=True)
modelSVM.fit(x_train,y_train)
score4=modelSVM.score(x_valid,y_valid)
predicted4=np.array(modelSVM.predict_proba(x_valid))
print("支持向量机log损失为 %f" %(log_loss(y_valid, predicted2)))
print("支持向量机mse为 %f" %(mean_squared_error(y_valid,predicted1)))
print('支持向量机准确率：', score2)