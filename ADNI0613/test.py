import os

import torch
import scipy.io as sio
import matplotlib.pyplot as plt
from Model import Model
from DataLoader.Data_Loader import data_flow



batch_size = 5
patch_size = 25
img_size=[121, 145, 121]


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load image
    img_path = '/data/home/wangzr/Projects/ADNI0613/ADNIpreprocess'
    data = sio.loadmat('/data/home/wangzr/Projects/ADNI0613/data_split/data.mat')
    sample_test = data['samples_test'].flatten()
    # flatten:展开为一维数组
    labels_test = data['labels_test'].flatten().reshape(380, 5)
    test_dataset = data_flow(img_path, sample_test, labels_test, img_size, patch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0,
                                               collate_fn=test_dataset.collate_fn)

    # create model
    model = Model.generate_model()
    model_weight_path = "./weights/model-100.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        sample_num = 0
        accu_num = torch.zeros(1).to(device)
        for step, data in enumerate(test_loader):
            images, labels = data
            sample_num += images.shape[1]
            print('sample_number', sample_num)
            output = model(images.to(device))
            predict_cla = torch.argmax(output,dim=1)[1]
            labels = torch.max(labels, dim=2)[1]
            accu_num += torch.eq(predict_cla, labels.to(device)).sum()
            print('accu_num', accu_num)
            print(output, labels)
            print('pred_classes', predict_cla)

        test_loader.desc = "acc: {:.3f}".format(accu_num.item() / sample_num)





if __name__ == '__main__':
    main()