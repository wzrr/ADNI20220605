import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

class data_flow(Dataset):

    def __init__(self,img_path, sample_name, sample_lbl, img_size, patch_size):
        self.img_path = img_path + '/{}'
        self.patch_size = patch_size
        self.sample_lbl = sample_lbl.tolist()
        self.sample_name = sample_name.tolist()
        self.input_shape = (1, patch_size, patch_size, patch_size)
        self.output_shape = (1, 5)
        self.margin = int(np.floor((patch_size - 1) / 2.0))
        self.xyznum_patches = (np.array(img_size) / patch_size).astype(int)
        self.num_patches = self.xyznum_patches[0] * self.xyznum_patches[1] * self.xyznum_patches[2]
        self.xyz = ((np.array(img_size) - (np.array(img_size) / patch_size).astype(int) * patch_size) / 2 + self.margin + 1).astype(
        int)

    def __len__(self):
        return len(self.sample_name)


    def __getitem__(self, item):
        self.img_dir = self.img_path.format(self.sample_name[item].strip())
        I = nib.load(self.img_dir)
        img = I.get_data()

        inputs = []
        for i_input in range(self.num_patches):
            inputs.append(np.zeros(self.input_shape, dtype='float32'))

        x_cor = self.xyz[0]
        i_patch = -1

        for xi_patch in range(self.xyznum_patches[0]):
            y_cor = self.xyz[1]
            for yi_patch in range(self.xyznum_patches[1]):
                z_cor = self.xyz[2]
                for zi_patch in range(self.xyznum_patches[2]):
                    img_patch = img[x_cor - self.margin: x_cor + self.margin + 1,
                            y_cor - self.margin: y_cor + self.margin + 1,
                            z_cor - self.margin: z_cor + self.margin + 1]

                    i_patch += 1
                    #if i_patch in (0,1,2,3,7,11,15,16,17,18,19,20,21,22,23,27,35，39,40,41,42,43,47,55,59,60,61,62,63,67,75,76,79):
                        #40,41,47,60,61,67,76
                        #img_patch=np.zeros((self.patch_size, self.patch_size, self.patch_size))
                    img_patch[np.isnan(img_patch)] = 0.0
                    #print('原始数据',i_patch,img_patch)
                    inputs[i_patch][0, :, :, :] = img_patch
                    z_cor += self.patch_size
                y_cor += self.patch_size
            x_cor += self.patch_size

        outputs = self.sample_lbl[item] * np.ones(self.output_shape, dtype='long')

        np.set_printoptions(precision=4)

        return torch.from_numpy(np.array(inputs)), outputs

    def collate_fn(self,batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        #
        #tuple：创建元组
        images, labels = tuple(zip(*batch))

        #扩维拼接
        images = torch.stack(images, dim=1)
        labels = torch.as_tensor(np.array(labels))
        torch.set_printoptions(precision=4)
        return images, labels


