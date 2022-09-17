#看一下之前的大作业

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

key=['ATSc2','ATSc3','ATSc5','BCUTc-1l','BCUTc-1h','VC-5','SdssC','minHBa','minHBint5','minHBint10','minsssN','maxssO','MAXDP','ETA_Shape_Y','MDEC-23','MDEC-33','MLFER_A','WTPT-4','XLogP']


data=pd.read_csv("aftergrey.csv",index_col=0)
feature=data[key]   #自变量

d=feature.corr()
print(d)

plt.subplots(figsize=(12, 12))
sns.heatmap(d, annot=False, vmax=1, square=True, cmap="Reds")
plt.show()