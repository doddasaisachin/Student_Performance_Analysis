import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv("./student-por.csv")

df.head()

df.shape

# df.info()

df.isna().sum()

col_to_be_del=[
    'reason','schoolsup','famsup',
    'paid','activities','higher',
    'romantic','famrel','Dalc','Walc'
]

df.head()

df.columns

df.drop(col_to_be_del,axis=1,inplace=True)

df.describe()

df['school'].dtype=='O'

def categorical_col(df):
    arr=[]
    for i in df.columns:
        if df[i].dtype=='O':
            arr.append(i)
    return arr

cat_cols=categorical_col(df)
# print(cat_cols)

# for i in cat_cols:
#     print(i)
#     print(df[i].value_counts())
#     print('\n')

# import seaborn as sns
import matplotlib.pyplot as plt

def numerical_col(df):
    arr=[]
    for i in df.columns:
        if df[i].dtype!='O':
            arr.append(i)
    return arr

num_cols=numerical_col(df)
# print(num_cols)

# for i in num_cols:
#     print(i)
#     print(df[i].value_counts())
#     print('\n')

# for i in num_cols:
#     plt.figure(figsize=(5,3))
#     sns.distplot(df[i],kde=True)
#     plt.show()

df_copy=df.copy()

from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()

# print(cat_cols)

for i in cat_cols:
    enc=LabelEncoder()
    df_copy[i]=enc.fit_transform(df_copy[i])

cat_cols_copy=categorical_col(df_copy)
print(cat_cols_copy)

df_copy.describe()

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

y=df_copy.pop('G3')

models={
    'AdaBoostRegressor':AdaBoostRegressor(),
    'ExtraTreesRegressor':ExtraTreesRegressor(),
    'GradientBoostingRegressor':GradientBoostingRegressor(),
    'RandomForestRegressor':RandomForestRegressor(),
    'LinearRegression':LinearRegression(),
    'SVR':SVR(),
    'KNeighborsRegressor':KNeighborsRegressor()
}

X=df_copy

xtrain,xtest,ytrain,ytest=train_test_split(X,y,random_state=50)

print(xtrain.shape,ytrain.shape,xtest.shape,ytest.shape)

from sklearn.metrics import r2_score,mean_squared_error

def best_model(ml_models,xtrain,xtest,ytrain,ytest):
    performance=dict()
    models=[]
    idx=0
    for key in ml_models.keys():
        arr=[]
        arr.append(idx)
        idx+=1
        model=ml_models[key]
        model.fit(xtrain,ytrain)
        y_pred=model.predict(xtest)
        score=r2_score(ytest,y_pred)
        arr.append(score)
        mse=mean_squared_error(ytest,y_pred)
        arr.append(mse)
        performance[key]=arr
        models.append(model)
    return performance,models

performance,Models= best_model(models,xtrain,xtest,ytrain,ytest)

performance

def find_best_model(performance,Models):
    arr=sorted(performance,key=lambda x:performance[x][1],reverse=True)
    key=arr[0]
    idx=performance[key][0]
    return Models[idx]

performance

Models

reg_model=find_best_model(performance,Models)
reg_model

from sklearn.model_selection import RandomizedSearchCV

param_dict={
    'n_estimators':[100,125,150],
    'criterion':['friedman_mse', 'squared_error', 'mse'],
    'max_depth':[2,3,4]
}

rscv_reg=RandomizedSearchCV(reg_model,param_dict,n_iter=10,cv=5,verbose=True,random_state=50)

rscv_reg.fit(xtrain,ytrain)

rscv_reg.best_params_

rscv_reg.best_estimator_

rscv_reg.best_score_

ypred=rscv_reg.predict(xtest)

len(ypred)

# plt.scatter(x=[i for i in range(1,len(ypred)+1)],y=ypred,c='purple')
# plt.scatter(x=[i for i in range(1,len(ypred)+1)],y=ytest,c='red')

# plt.plot(np.array(ypred)-np.array(ytest))



import pickle

pickle.dump(rscv_reg,open('reg_model.pkl','wb'))
