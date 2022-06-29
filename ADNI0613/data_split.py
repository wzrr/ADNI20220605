import os
import numpy as np
import scipy.io as sio
from random import sample

img_path = "/data/home/wangzr/Projects/ADNI0613/ADNIpreprocess/"
resultpath = '/data/home/wangzr/Projects/ADNI0613/data_split/data.mat'
sample_name = []
labels = []

a=np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
print(a)
print(np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]))
b=a.take(indices=[1,2],axis=0)
print(b)
for img in os.listdir(img_path + 'CN'):
        #listdir：得到该文件夹目录下的文件
        sample_name.append('CN/' + img)
        labels.append([1,0,0,0,0])
for img in os.listdir(img_path + 'AD'):
        sample_name.append('AD/' + img)
        labels.append([0,1,0,0,0])
for img in os.listdir(img_path + 'EMCI'):
        sample_name.append('EMCI/' + img)
        labels.append([0,0,1,0,0])
for img in os.listdir(img_path + 'LMCI'):
        sample_name.append('LMCI/' + img)
        labels.append([0,0,0,1,0])
for img in os.listdir(img_path + 'SMC'):
        sample_name.append('SMC/' + img)
        labels.append([0,0,0,0,1])

print(labels)
sample_name = np.array(sample_name)
labels = np.array(labels)
print(labels)
permut = np.random.permutation(len(sample_name))
print(permut)
#返回0-样本个数-1的随机排列数列
np.take(sample_name, permut, out=sample_name)
print(labels)
c=labels.take(indices=permut, axis=0)
labels=c
print(labels)
#sample_name和labels数组分别按permut洗牌
print(len(labels))

CNlist = labels[: , 0]
CN_list= np.where( CNlist == 1)[0]
ADlist = labels[: , 1]
AD_list = np.where( ADlist == 1)[0]
EMCIlist = labels[: , 2]
EMCI_list = np.where( EMCIlist == 1)[0]
LMCIlist = labels[: , 3]
LMCI_list = np.where( LMCIlist == 1)[0]
SMClist = labels[: , 4]
SMC_list = np.where( SMClist == 1)[0]
#返回洗牌后每个组的下标

AD_test = sample(list(AD_list), round(len(AD_list)/5))
print(AD_test)
CN_test = sample(list(CN_list), round(len(CN_list)/5))
EMCI_test = sample(list(EMCI_list), round(len(EMCI_list)/5))
LMCI_test = sample(list(LMCI_list), round(len(LMCI_list)/5))
SMC_test = sample(list(SMC_list), round(len(SMC_list)/5))

#随机抽样各五分之一
test_list = AD_test + CN_test + EMCI_test + LMCI_test + SMC_test
test_list = sorted(test_list)
train_list = list(set(range(len(sample_name))).difference(set(test_list)))
#train_list为sample与test_list的差集

samples_train = sample_name[train_list]
labels_train = labels[train_list]
samples_test = sample_name[test_list]
labels_test = labels[test_list]

print(samples_train,labels_train,samples_test,labels_test)
print(len(samples_train),len(labels_train),len(samples_test),len(labels_test))


sio.savemat(resultpath, {"samples_train": samples_train,"samples_test": samples_test,"labels_train": labels_train,"labels_test": labels_test})