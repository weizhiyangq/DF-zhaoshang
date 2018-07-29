# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 12:42:52 2018

@author: YWZQ
"""

import pandas as pd
import numpy as np
import time 
from pandas import DataFrame

# 读取个人信息
train_agg = pd.read_csv(r'.\data\train_agg.csv',sep='\t')
test_agg = pd.read_csv(r'.\data\test_agg.csv',sep='\t')
agg = pd.concat([train_agg,test_agg],copy=False)

# 用户唯一标识
train_flg = pd.read_csv(r'.\data\train_flg.csv',sep='\t')
test_flg = pd.read_csv(r'.\data\submit_sample.csv',sep='\t')
test_flg['FLAG'] = -1
del test_flg['RST']
flg = pd.concat([train_flg,test_flg])
#data = pd.merge(agg_normal,flg,on=['USRID'],how='left',copy=False)
data = pd.merge(agg,flg,on=['USRID'],how='left',copy=False)

# 日志信息
train_log = pd.read_csv(r'.\data\train_log.csv',sep='\t')
test_log = pd.read_csv(r'.\data\test_log.csv',sep='\t')
log = pd.concat([train_log,test_log],copy=False)

# =============================================================================
# 用户点击次数
# =============================================================================
log_usrid_count = log[['USRID']]
log_usrid_count['count'] = 1
log_usrid_count = log_usrid_count.groupby(['USRID'],as_index=False)['count'].sum()
data = pd.merge(data,log_usrid_count,on=['USRID'],how='left',copy=False)


# =============================================================================
#统计各个用户点击top30热门的模块次数
# =============================================================================

evtlbl_list=log['EVT_LBL'].unique()  #查看有多少种类模块
print(len(evtlbl_list))
log['ci']=1
evt_group=log['ci'].groupby([log['USRID'],log['EVT_LBL']]).sum()
evt_group_unstack=evt_group.unstack()
evt_group_unstack.fillna(0,inplace=True)
evt_group=evt_group_unstack.reset_index('USRID')  #每个用户对每个模块点击次数
all_module_count=DataFrame(evt_group_unstack.sum(axis=1),columns=['all_module_count'])  #各个用户点击所有模块的次数
all_module_count=all_module_count.reset_index('USRID')
print(all_module_count)
useid=evt_group.loc[:,['USRID']]
evt_group.drop(['USRID'],axis=1,inplace=True)
num=evt_group.sum()
num_sort=DataFrame(num.sort_values(ascending=False)).reset_index()
num_sort.columns=['dianpu','count']
print(num_sort[:30])
hotshop=num_sort[:30]['dianpu'].tolist()
hotshop_group=evt_group.loc[:,hotshop]
hotshop_group_num=DataFrame(hotshop_group.sum(axis=1))
hotshop_group=pd.concat([useid,hotshop_group_num],axis=1)
hotshop_group.columns=['USRID','hotcount']
print(hotshop_group)

data = pd.merge(data,hotshop_group,on=['USRID'],how='left',copy=False)

# =============================================================================

#离4月1最近的一次点击是什么时候、最远是什么时候
# =============================================================================

time_last=DataFrame()
time_last['OCC_TIM']=log['OCC_TIM'].map(lambda x:x[8]+x[9])
time_last['OCC_TIM']=log['OCC_TIM'].astype(int)

time_last['OCC_TIM']=log['OCC_TIM'].map(lambda x:32-x)
near_time_group=DataFrame(log['OCC_TIM'].groupby([log['USRID']]).min()).reset_index('USRID')
far_time_group=DataFrame(log['OCC_TIM'].groupby([log['USRID']]).max()).reset_index('USRID')
near_time_group.columns=['USRID','time_dist']
far_time_group.columns=['USRID','time_dist_far']
print(near_time_group)
time_distance=pd.merge(near_time_group,far_time_group,on='USRID',how='left')
data = pd.merge(data,time_distance,on=['USRID'],how='left',copy=False)


# =============================================================================
# 前一个月统计，点击为1，没有为0
# 统计前2,3,4,5,6,7天的点击量 0.8579184451380367(7天提升) 0.8572315952056823（5天下降） 
# 加上agg --> 0.8577258763466098提升一点点
# =============================================================================
log_time = log[['USRID','OCC_TIM']]
log_time['day'] = log['OCC_TIM'].apply(lambda x:int(x[8:10]))
log_time = log_time.drop(['OCC_TIM'],axis=1)
days = pd.get_dummies(log_time['day'])
days.columns = ['mooth_day'+str(i+1) for i in range(days.shape[1])]
log_time = pd.concat([log_time[['USRID']],days],axis=1)
log_time = log_time.groupby(['USRID'],as_index=False).sum()
log_time['front_2'] = log_time['mooth_day31']+log_time['mooth_day30']
for i in range(3,8):
    col = 'front_'+str(i)
    col_front = 'front_'+str(i-1)
    col_add = 'mooth_day'+str(32-i)
    log_time[col] = log_time[col_front] + log_time[col_add]  
data = pd.merge(data,log_time,on=['USRID'],how='left',copy=False)

# =============================================================================
# 点击模块名称均为数字编码（形如231-145-18），代表了点击模块的三个级别（如饭票-代金券-门店详情）
# =============================================================================
log_mode = log[['USRID','EVT_LBL']]
log_mode['EVT_LBL_1'] = log_mode['EVT_LBL'].apply(lambda x:int(x.split('-')[0]))
EVT = pd.get_dummies(log_mode['EVT_LBL_1'])
EVT.columns = ['level1_'+str(i+1) for i in range(EVT.shape[1])]
log_mode = pd.concat([log_mode[['USRID']],EVT],axis=1)
log_mode = log_mode.groupby(['USRID'],as_index=False).sum()
data = pd.merge(data,log_mode,on=['USRID'],how='left',copy=False)
log_mode = log[['USRID','EVT_LBL']]
log_mode['EVT_LBL_2'] = log_mode['EVT_LBL'].apply(lambda x:int(x.split('-')[1]))
EVT = pd.get_dummies(log_mode['EVT_LBL_2'])
EVT.columns = ['level2_'+str(i+1) for i in range(EVT.shape[1])]
log_mode['EVT_LBL_3'] = log_mode['EVT_LBL'].apply(lambda x:int(x.split('-')[2]))
log_mode = pd.concat([log_mode[['USRID']],EVT],axis=1)
log_mode = log_mode.groupby(['USRID'],as_index=False).sum()
data = pd.merge(data,log_mode,on=['USRID'],how='left',copy=False)

# =============================================================================
#对模块三级进行分切，接着进行分箱统计
# =============================================================================

model_split=log_mode.loc[:,['USRID','EVT_LBL_0','EVT_LBL_1','EVT_LBL_2']]
def get_level(df,level=0):
    label=''
    if level==0:
        label='EVT_LBL_0'
    elif level==1:
        label='EVT_LBL_1'
    else :
        label='EVT_LBL_2'
    cut_label=pd.get_dummies(pd.cut(df[label],5))
    level_cut=pd.concat([df['USRID'],cut_label],axis=1)
    level_count=level_cut.groupby('USRID',as_index=False).sum()
    print(level_count.shape)
    return level_count
level_count_0=get_level(model_split,0)
level_count_1=get_level(model_split,1)
level_count=pd.merge(level_count_0,level_count_1,how='left',on='USRID')
data = pd.merge(data,level_count,on=['USRID'],how='left',copy=False)


# =============================================================================
# 浏览类型
# =============================================================================
log_type = log[['USRID','TCH_TYP']]  
types = pd.get_dummies(log_type['TCH_TYP']) 
types.columns = ['APP','H5']
log_type = pd.concat([log_type[['USRID']],types],axis=1)
log_type = log_type.groupby(['USRID'],as_index=False).sum()
data = pd.merge(data,log_type,on=['USRID'],how='left',copy=False)

#=============================================================================
# 这个部分将时间转化为秒，之后计算用户下一次的时间差特征
# =============================================================================
log['OCC_TIM'] = log['OCC_TIM'].apply(lambda x:time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
log = log.sort_values(['USRID','OCC_TIM'])
log['next_time'] = log.groupby(['USRID'])['OCC_TIM'].diff(-1).apply(np.abs)
log = log.groupby(['USRID'],as_index=False)['next_time'].agg({
        'next_time_mean':np.mean,
        'next_time_std':np.std,
        'next_time_min':np.min,
        'next_time_max':np.max
})
data = pd.merge(data,log,on=['USRID'],how='left',copy=False)

print(data.shape)
weather_has_app=pd.read_csv(r'E:\datafountain\zhaoshang\feature_clean\wether_app.csv')

data=pd.merge(data,weather_has_app,how='left',on='USRID')

data['has_app'].fillna(0,inplace=True)

print(data.shape)


data.to_csv(r'.\feature\feature.csv',index=False,sep='\t')
