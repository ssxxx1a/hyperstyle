import os
import time
import pickle
import random
from tkinter.tix import MAIN
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from utils import aus_utils

class AusDataset(Dataset):
    def __init__(self, opt):
        super(AusDataset, self).__init__()
        self.opt = opt
        self.transform = transforms.Compose([transforms.Resize((256, 256)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
                                            )
        self.read_dataset(self.opt)
    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):
        assert (idx < self.len_data)
        real_img = None
        real_cond = None
        real_img_path = None
        while real_img is None or real_cond is None:
            sample_id = self.get_id(idx)
            real_img, real_img_path = self.get_img_by_id(idx)
            real_cond = self.get_cond_by_id(idx)
            if real_img is None:
                print('error reading image %s, skipping sample' % real_img_path)
                idx = random.randint(0, self.len_data - 1)
            if real_cond is None:
                print('error reading aus %s, skipping sample' % sample_id)
                idx = random.randint(0, self.len_data - 1)
        desired_img, desired_cond, noise = self.generate_random_cond()
        real_img = self.transform(Image.fromarray(real_img))
        desired_img = self.transform(Image.fromarray(desired_img))

        # pack data
        sample = {'real_img': real_img,
                  'real_cond': real_cond,
                  'desired_img': desired_img,
                  'desired_cond': desired_cond,
                  'real_img_path': real_img_path
                  }
        return sample

    def read_dataset(self, data_type='csv'):
        self.root = self.opt.root_path
        annotations_file = self.opt.data_train
        pkl_path = os.path.join(self.root, annotations_file)
        print('load data path:', pkl_path)
        # self._info = self._read_pkl(pkl_path)
        if data_type == 'csv':
            self.info = self.read_csv(self.opt.csv_file)
        else:
            self.info = self.read_pkl(pkl_path)
        self.image_size = self.opt.output_size
        # dataset size
        self.len_data = len(self.info)
        print('dataset size:', self.len_data)

    def read_pkl(self, file_path):
        assert os.path.exists(file_path) and file_path.endswith(
            '.pkl'), 'Read pkl file error. Cannot open %s' % file_path
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def read_csv(self, csv_path):
        assert os.path.exists(csv_path) and csv_path.endswith(
            '.csv'), 'Read pkl file error. Cannot open %s' % csv_path
        data = []
        with open(csv_path, 'r') as f:
            for d in f:
                path_au = d.rstrip('\n').split(',')
                data.append({'file_path': str(path_au[0]),
                             'aus': np.array([float(l) for l in path_au[1:]])})

        return data

    def get_id(self, idx):
        id = self.info[idx]['file_path']
        return os.path.splitext(id)[0]

    def get_cond_by_id(self, idx):
        cond = None
        if idx < self.len_data:
            cond = np.round(self.info[idx]['aus'] / 5.0, 3)  # 保留3位置
        return cond

    def get_img_by_id(self, idx):
        if idx < self.len_data:
            img_path = self.info[idx]['file_path']  # 修改后不需要join
            re_img_path = os.path.join(self.root,"new_refined/",img_path.split('/')[-1])
            # if self.opt.data_train == 'best_data.pkl':
            #     if self.opt.ip == 'v100':
            #         re_img_path = self.root + img_path.split('/')[-1]
            # else:
            #     raise ValueError('pkl isnt best_data.pkl')
            img = aus_utils.read_cv2_img(re_img_path)
            return img, self.info[idx]['file_path']
        else:
            print('You input idx： ', idx)
            return None, None

    def generate_random_cond(self):
        cond = None
        rand_sample_id = -1

        while cond is None:
            rand_sample_id = random.randint(0, self.len_data - 1)
            cond = self.get_cond_by_id(rand_sample_id)
        img, _ = self.get_img_by_id(rand_sample_id)
        # noise = np.random.uniform(-0.1, 0.1, cond.shape)
        if img is None:
            img, cond, _ = self.generate_random_cond()
        # cond += noise
        return img, cond, _
