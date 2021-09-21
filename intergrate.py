# -*- coding:utf-8 -*-
# @Author : Dummerfu
# @Contact : https://github.com/dummerchen 
# @Time : 2021/8/14 22:33
import glob

import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import random

mp={'A': '0',
 'B': '1',
 'C': '2',
 'D': '3',
 'E': '4',
 'F': '5',
 'G': '6',
 'H': '7',
 'I': '8',
 'J': '9',
 'K': '10',
 'N': '11',
 'O': '12',
 'P': '13',
 'Q': '14',
 'R': '15',
 'S': '16',
 'T': '17',
 'U': '18',
 'V': '19',
 'X': '20',
 'Z': '21'}
rmp={
    '0': 'A',
 '1': 'B',
 '2': 'C',
 '3': 'D',
 '4': 'E',
 '5': 'F',
 '6': 'G',
 '7': 'H',
 '8': 'I',
 '9': 'J',
 '10': 'K',
 '11': 'N',
 '12': 'O',
 '13': 'P',
 '14': 'Q',
 '15': 'R',
 '16': 'S',
 '17': 'T',
 '18': 'U',
 '19': 'V',
 '20': 'X',
 '21': 'Z'
}
random.seed(1314)

def to_onehot(x):
    label = np.zeros(22)
    label[eval(mp[x])] = 1
    return label


def get_ans(x:pd.Series,num:int,inconfident_label:int):
    nlabel = np.zeros(22)
    for i in range(num):
        if i == inconfident_label:
            continue
        try:
            nlabel += x[str(i)]
        except:
            pass
    return nlabel

def check(label:np.array):
    res = np.max(label)
    pos = []
    for num,i in enumerate(label):
        if i == res:
            pos.append(num)
    return pos

def func1(x:pd.Series,num,confident_label,inconfident_list:list):
    # 提取置信度高的预测的标签
    inconfident_label_list=[-1]+inconfident_list
    ans=None
    pos=[]
    temp_x=x
    for i in inconfident_label_list:
        try:
            temp_x = temp_x.drop(str(i))
        except:
            pass
        label = get_ans(temp_x, num, i)

        pos=check(label)

        if len(pos)>1:
            continue
        else:
            ans=pos[0]
            break

    if ans==None:
        ans=random.choice(pos)
    x['res1_label']=rmp[str(int(ans))]
    return x

def func2(x:pd.Series,num,weight):
    # 提取置信度高的预测的标签
    nlabel = np.zeros(22)
    for i in range(num):
        nlabel += np.array(x[str(i)])*weight[i]
    res = np.max(nlabel)
    pos = []
    for num, i in enumerate(nlabel):
        if i == res:
            pos.append(num)
    if len(pos)>1:
        ans = random.choice(pos)
    else:
        ans = pos[0]
    x['res2_label']=rmp[str(int(ans))]
    return x


def predict(paths:list,confident_label,inconfident_list,weight=None):
    '''
        输入权重路径，tuple
        is_weight: 是否使用加权，默认投票
    '''
    
    raw_data=pd.read_csv('./testzh0.csv')
    num=len(paths)
    for i,pth in enumerate(paths):
        data=pd.read_csv(pth)
        raw_data['label_{}'.format(name[i])]=data['label']
        if weight.any()==None:
            data['label']=data['label'].apply(to_onehot)
        else:
            # data['label'] = data['label'].apply(lambda x:eval(x)*weight[i])
            data['label'] = data['label'].apply(lambda x:eval(x))


        data['label']=np.array(data['label'])
        raw_data[str(i)]=data['label']
    # raw_data=raw_data.apply(func1,args=(num,confident_label,inconfident_list),axis=1)
    raw_data=raw_data.apply(func2,args=(num,weight),axis=1)

    for i in range(num):
        raw_data.drop(labels=str(i),inplace=True,axis=1)

    raw_data.to_csv('./result/intergrate_test2.csv',index=None)

if __name__ == '__main__':
    # npy权重放置在./prov_test/下
    name = glob.glob('./prov_test/*.npy')
    name = [i.replace('./prov_test\\', '').replace('.npy', '') for i in name]
    # 平均加权融合
    weight=np.ones(len(name))

    data_path=['./prov_test/{}.csv'.format(i) for i in name]

    predict(data_path,confident_label=3,inconfident_list=[0],weight=weight)