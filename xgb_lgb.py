# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 14:18:26 2018

@author: YWZQ
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from pandas import DataFrame
from pandas import merge

data=pd.read_csv(r'.\feature\feature.csv',sep='\t')


data_not_hasto_stad=data.iloc[:,30:32]
data_hasto_stad_1=data.iloc[:,0:30]
data_hasto_stad_2=data.iloc[:,32:]
data_hasto_stad=pd.concat([data_hasto_stad_1,data_hasto_stad_2],axis=1)

#data_hasto_stad=(data_hasto_stad-data_hasto_stad.min())/(data_hasto_stad.max()-data_hasto_stad.min())

data_hasto_stad=(data_hasto_stad-data_hasto_stad.min())/(data_hasto_stad.max()-data_hasto_stad.min())

data=pd.concat([data_not_hasto_stad,data_hasto_stad],axis=1)
#print("data_max after stand:\n")
#print(data.max())

#data = data.drop(['Unnamed: 0'],axis=1)
#data = pd.read_csv('feature_analyze_drop.csv',sep='\t')
train = data[data['FLAG']!=-1]
test = data[data['FLAG']==-1]

train_userid = train.pop('USRID')
y = train.pop('FLAG')
col = train.columns
X = train[col].values
X_pd=train[col]

test_userid = test.pop('USRID')
test_y = test.pop('FLAG')
test = test[col].values
N = 5
skf = StratifiedKFold(n_splits=N,shuffle=True,random_state=42)

lgb_cv = []
lgb_pre = []
xgb_cv = []
xgb_pre = []
mean_cv = []
mean_pre = []
mean_cv2 = []
mean_pre2 = []
mean_cv3 = []
mean_pre3 = []
mean_cv4 = []
mean_pre4 = []

all_val_pre=DataFrame()

for train_in,test_in in skf.split(X,y):
    X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]
    X_Pd_val=X_pd.iloc[test_in,:]  #将对这个进行预测，作为stacking的输入，stacking文件不包含于此文件

#    ########################## lightgbm ###############################
#    light_params = {
#            'boosting_type': 'gbdt',
#            'objective': 'binary',
#            'metric': {'auc'},
#            'num_leaves': 32,
#            'learning_rate': 0.01,
#            'feature_fraction': 0.9,
#            'bagging_fraction': 0.8,
#            'bagging_freq': 5,
#            'verbose': 0,
#            'lambda_l2' : 20,
#            'min_child_weight': 9,
#        }
#    lgb_train = lgb.Dataset(X_train, y_train)
#    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
#    print('Lightgbm Start training...')
#    gbm = lgb.train(light_params,
#                    lgb_train,
#                    num_boost_round=5000,
#                    valid_sets=lgb_eval,
#                    verbose_eval=500,
#                    early_stopping_rounds=300)
#    print('Lightgbm Start predicting...')
#    lgb_val_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
#    lgb_cv.append(roc_auc_score(y_test,lgb_val_pred))
#    lgb_pred = gbm.predict(test, num_iteration=gbm.best_iteration)
#    lgb_pre.append(lgb_pred)
    
    ########################## xgboost ###############################
    xgboost_params = {'booster': 'gbtree',
                      'objective':'binary:logistic',
                      'eta': 0.01,
                      'max_depth': 5, 
                      'colsample_bytree': 0.7,
                      'subsample': 0.7,
                      'min_child_weight': 9, 
                      'silent':1,
                      'eval_metric':'auc',
                      'lambda' : 20,
                      }
    print('XGBoost Start training...')
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    watchlist = [(dtrain,'train'),(dtest,'val')] 
    xgb_model = xgb.train(xgboost_params, dtrain,5000,evals=watchlist,verbose_eval=500,early_stopping_rounds=280)
    print('XGBoost Start predicting...')
    dvali = xgb.DMatrix(X_test)
    xgb_val_pred = xgb_model.predict(dvali,ntree_limit=xgb_model.best_ntree_limit)
    
    X_Pd_val['pre_y']=xgb_val_pred
    all_val_pre=pd.concat([all_val_pre,X_Pd_val])
    print(all_val_pre.head())
    
    xgb_cv.append(roc_auc_score(y_test,xgb_val_pred))
    
    dx=xgb.DMatrix(X,label=y)
    xgb_model_all=xgb.train(xgboost_params, dx,num_boost_round=xgb_model.best_ntree_limit,evals=[(dx,'dx')],verbose_eval=500,early_stopping_rounds=280)
    
    dfinal = xgb.DMatrix(test)
    xgb_pred = xgb_model_all.predict(dfinal,ntree_limit=xgb_model_all.best_ntree_limit)
    xgb_pre.append(xgb_pred)

print('val_pre:\n')   
print(all_val_pre['pre_y'])  
all_val_pre.to_csv(r'.\stacking_feature.csv',index=False,sep='\t')  

s_xgb = 0
for i in xgb_pre:
    s_xgb = s_xgb + i

s_xgb = s_xgb /N
res_xgb = pd.DataFrame()
res_xgb['USRID'] = list(test_userid.values)
res_xgb['RST'] = list(s_xgb)
print('xgboost_cv',np.mean(xgb_cv))
test_id=pd.read_csv(r'.\data\submit_sample.csv',sep='\t')
test_id.drop(['RST'],axis=1,inplace=True)
res_xgb=pd.merge(test_id,res_xgb,how='left',on='USRID')
print(res_xgb)


res_xgb.to_csv(r'.\result\result.csv',index=False,sep='\t')
