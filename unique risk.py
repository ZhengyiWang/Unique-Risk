# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:15:23 2021

@author: dell
"""

#https://www.zhihu.com/question/28641663


###########1.导入数据部分##########
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

unique_risk = pd.read_csv('unique_risk.csv')

unique_risk['审核结果']=unique_risk['审核结果'].map({'发行股份购买资产获无条件通过':1,'发行股份购买资产获有条件通过':0})



# 抽取 30% 的数据作为测试集，其余作为训练集
from sklearn.model_selection import train_test_split
train, test = train_test_split(unique_risk, test_size = 0.3)

# 抽取特征选择的数值作为训练和测试数据
#后期考虑是否对行业、审批结果进行编码，
train_X = train.drop(labels=['公司名称',"Rc"], axis=1)
train_y=train['Rc']
test_X= train.drop(labels=['公司名称',"Rc"], axis=1)
test_y =train['Rc']


from sklearn.preprocessing import StandardScaler
# 采用 Z-Score 规范化数据，保证每个特征维度的数据均值为 0，方差为 1
ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.transform(test_X)



import seaborn as sns
import matplotlib.style as style
# 选用一个干净的主题
style.use('fivethirtyeight')
sns.heatmap(unique_risk.corr())



from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

#GBDT作为基模型的特征选择
SelectFromModel(GradientBoostingClassifier()).fit_transform(train_X, train_y)



from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(train_X, train_y)
model = SelectFromModel(lsvc, prefit=True)
train_X_new = model.transform(train_X)
train_X_new.shape


from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

#加载波士顿房价作为数据集
boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

names=unique_risk.columns.values.tolist()

#n_estimators为森林中树木数量，max_depth树的最大深度
rf = RandomForestRegressor(n_estimators=20, max_depth=4)
scores = []
for i in range(train_X.shape[1]):
    #每次选择一个特征，进行交叉验证，训练集和测试集为7:3的比例进行分配，
    #ShuffleSplit()函数用于随机抽样（数据集总数，迭代次数，test所占比例）
    score = cross_val_score(rf, train_X[:, i:i+1], train_y, scoring="r2",
                               cv=ShuffleSplit(len(train_X), 3, .3))
    scores.append((round(np.mean(score), 3), names[i]))

#打印出各个特征所对应的得分
print(sorted(scores, reverse=True))





###########2.回归部分##########
def try_different_method(model):
    model.fit(train_X,train_y)
    score = model.score(test_X, test_y)
    result = model.predict(test_X)
    plt.figure()
    plt.plot(np.arange(len(result)), test_y,'go-',label='true value')
    plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
    plt.title('score: %f'%score)
    plt.legend()
    plt.show()








###########3.具体方法选择##########
####3.1决策树回归####
from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
####3.2线性回归####
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()
####3.3SVM回归####
from sklearn import svm
model_SVR = svm.SVR(C=1.0, epsilon=0.2)
####3.4KNN回归####
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
####3.5随机森林回归####
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树
####3.6Adaboost回归####
from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
####3.7GBRT回归####
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
####3.8Bagging回归####
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor()
####3.9ExtraTree极端随机树回归####
from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()
####3.10 MLPRegressor神经网络回归####
from sklearn.neural_network import MLPRegressor
model_MLPRegressor=MLPRegressor()

###########4.具体方法调用部分##########
try_different_method(model_SVR)

try_different_method(model_AdaBoostRegressor)

try_different_method(model_ExtraTreeRegressor)

try_different_method(model_DecisionTreeRegressor)

try_different_method(model_LinearRegression)

try_different_method(model_MLPRegressor)