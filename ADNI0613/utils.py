import os
import sys
import json
import pickle
import random
import scipy.io as sio
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

img_path = '/data/home/wangzr/Projects/ADNI0613/ADNIpreprocess'
#%%
data = sio.loadmat('/data/home/wangzr/Projects/ADNI0613/data_split/data.mat')
sample_name = data['samples_train'].flatten()
#flatten:展开为一维数组
labels = data['labels_train'].flatten().reshape(1520,5)

def read_split_data(i):
    #random.seed(0)  # 保证随机结果可复现
        # 20% training samples as the validation set
    valid_list = range(len(sample_name) // 5 * i , len(sample_name) // 5 * (i + 1)+1)
    train_list = list(set(range(len(sample_name))).difference(set(valid_list)))
    labels_train = labels[train_list]
    labels_valid = labels[valid_list]
    samples_train = sample_name[train_list]
    samples_valid = sample_name[valid_list]


    return samples_train, labels_train, samples_valid, labels_valid

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        labels = torch.max(labels, dim=2)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        print(pred, labels)
        labels_flatten = labels.flatten()
        loss = loss_function(pred, labels_flatten.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[1]
        print('sample_number',sample_num)

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        labels = torch.max(labels, dim=2)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        print('accu_num',accu_num)
        print(pred,labels)
        print('pred_classes',pred_classes)

        labels_flatten = labels.flatten()

        loss = loss_function(pred, labels_flatten.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num