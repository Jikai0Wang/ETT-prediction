import os
import json
import pandas as pd
import numpy as np
import random

#读取原始数据
train_df = pd.read_csv('./raw_data/train_set.csv')
val_df = pd.read_csv('./raw_data/validation_set.csv')
test_df = pd.read_csv('./raw_data/test_set.csv')
train=np.array(train_df)
val=np.array(val_df)
test=np.array(test_df)

#设定滑动窗口大小
in_window=96
out_window=336

# #按6：2：2切分成train,val,test
# train=train_data[:round(raw_data.shape[0]*0.6),:]
# val=raw_data[round(raw_data.shape[0]*0.6):round(raw_data.shape[0]*0.8),:]
# test=raw_data[round(raw_data.shape[0]*0.8):,:]

#为后续进行数据标准化,根据训练集计算每个特征的均值和方差
features=train[:,1:]
features = features.astype(float)
mean_values = np.mean(features, axis=0).tolist()
std_values = np.std(features, axis=0).tolist()
d={"mean": mean_values, "std": std_values}

#滑动窗口采样数据
sampled_train=[]
for index in range(train.shape[0]-in_window-out_window+1):
    sampled_train.append({"input":train[index:index+in_window,:].tolist(),"label":train[index+in_window:index+in_window+out_window,:].tolist()})
sampled_val=[]
for index in range(val.shape[0]-in_window-out_window+1):
    sampled_val.append({"input":val[index:index+in_window,:].tolist(),"label":val[index+in_window:index+in_window+out_window,:].tolist()})
sampled_test=[]
for index in range(test.shape[0]-in_window-out_window+1):
    sampled_test.append({"input":test[index:index+in_window,:].tolist(),"label":test[index+in_window:index+in_window+out_window,:].tolist()})

#打乱数据
random.shuffle(sampled_train)
random.shuffle(sampled_val)
#random.shuffle(sampled_test)

#保存数据及各特征均值方差
output_dir="processed_data_{}_{}".format(in_window,out_window)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(output_dir + '/meam_std.json', mode='w', encoding='utf-8') as f0:
    json.dump(d, f0)
with open(output_dir+'/train.json', mode='w', encoding='utf-8') as f1:
    json.dump(sampled_train, f1)
with open(output_dir + '/val.json', mode='w', encoding='utf-8') as f2:
    json.dump(sampled_val, f2)
with open(output_dir + '/test.json', mode='w', encoding='utf-8') as f3:
    json.dump(sampled_test, f3)
