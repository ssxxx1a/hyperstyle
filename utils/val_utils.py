import torch
from e4e.e4e import eFe
import os
from data.dataset_aus import AusDataset
from utils import util
import pickle
from torchvision import transforms
from PIL import Image
import random
import glob
from face_recognition import face_encodings, compare_faces, face_distance, load_image_file, face_locations
import csv
# for tsne
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import argparse
import collections


def vaild_transfer(model, real_img, desired_img, in_au, de_au, save_path, save_g_only=False, return_w=False,
                   model_name='ours', fg_model=None, fg_img=None, use_step_grow=None):
    # 全都是tensor
    '''
    Args:
        model: 模型
        real_img: 真实图像
        desired_img: 期望图像
        in_au: 真实au值
        de_au: 期望au值s
        save_path: 保存的路径
        save_g_only: 是否只保留生成图
        return_w: 是否返回w向量
    Returns:
        可选返回w向量
    '''
    bs = real_img.size(0)
    if bs >= 8:
        nrow = 8
    else:
        nrow = bs
    sub_au = (de_au - in_au).float()
    de_au = de_au.float()
    # sub_au = torch.cat((sub_au[:, :6], sub_au[:, 7:]), dim=1)
    with torch.no_grad():
        if model_name == 'ours':
            with torch.no_grad():
                w_a, F_i = model.encoder(real_img, isfusion=True)
                '''mark'''
                w, g_fake_img = model.G(w_a, F_i, de_au.float(), in_au.float())

            if fg_model:
                if not use_step_grow:
                    fg_res = fg_model.forward(fg_img, sub_au)
                else:
                    # 逐步变化
                    # 因为只给一个au 0～1的变化，相对于原图就是其他维度给0，指定维度给值，所以是以下的做法
                    fg_res = fg_model.forward(fg_img, sub_au)
                fg_r = []
                for i in range(fg_res.size(0)):
                    fg_r.append(util.to_rgb_r(fg_res[i].cpu()).cuda())
                fg_res = torch.stack(fg_r, dim=0)
        else:
            g_fake_img = model.forward(real_img, sub_au)

    if not save_g_only:
        if fg_model:
            util.save_tensor_img(torch.cat([real_img[:8], desired_img[:8], g_fake_img[:8], fg_res[:8]], dim=0),
                                 save_path, nrow=nrow)

        else:
            util.save_tensor_img(torch.cat([real_img[:8], desired_img[:8], g_fake_img[:8]], dim=0),
                                 save_path, nrow=nrow)
    else:

        util.save_tensor_img(torch.cat([g_fake_img[:8]], dim=0),
                             save_path, nrow=nrow)
    if return_w:
        return w


def init_data(opts, val_path='/raid/hzhang/AffectNet/vaild/front2/'):
    train_dataset = AusDataset(opts)
    vaild_dataset = {
        'Neutral': [],
        'Happy': [],
        'Sad': [],
        'Surprise': [],
        'Fear': [],
        'Disgust': [],
        'Anger': [],
        'Contempt': [],
    }

    for root, dirs, files in os.walk(val_path):
        for file in files:
            express = root.split('/')[-1]
            vaild_dataset[express].append(file)
    with open(os.path.join(opts.root_path, 'best_data_val.pkl'), 'rb') as f:
        data = pickle.load(f)
    # print(data)
    au_map = {}
    for batch in data:
        au_map[batch['file_path'].split('/')[-1]] = batch['aus']

    return train_dataset, val_path, vaild_dataset, au_map


def image2tensor(transform, val_path, image_list, au_map, bs):
    '''
    Args:
    transform: transform类型
        val_path: 保存路径
        image_list: 转化的图像list
        bs: bs
    Returns:
    torch，(n,image_size)
    '''
    assert bs <= len(image_list)
    res = []
    for i in range(bs):
        if isinstance(au_map[image_list[i]], int):
            continue
        img = Image.fromarray(util.read_cv2_img(os.path.join(val_path, image_list[i])))
        res.append(transform(img))
    return torch.stack(res, dim=0)


def aus_from_map(au_map, image_list):
    '''
    Args:
        au_map: 文件名与au的映射map
        image_list: 图像的list
    Returns:
        torch :(n,17)
    '''
    res = []
    for i in image_list:
        if isinstance(au_map[i], int):
            continue
        res.append(torch.from_numpy(au_map[i]) / 5.0)
    return torch.stack(res, dim=0)
def Seq_contrs_gen(model, t, list1, list2, t_start, count, total, vaild_dataset, au_map, save_g_only=False,
                   save_de_au=False, model_name='ours', val_path='', save_path='analysis/res', fg_model=None,
                   use_step_grow=None, ):
    if not os.path.exists(os.path.join(save_path)):
        os.makedirs(os.path.join(save_path))
    if len(t) > 1:
        t1 = t[0]
        t2 = t[1]
    else:
        t1 = t[0]
    start = t_start
    '''
    重复迭代类别
    '''
    for e1 in list1:
        for e2 in list2:
            if e2 == e1:
                continue
            ori_type = e1
            target_type = e2
            if not total:
                total = min(len(vaild_dataset[ori_type]), len(vaild_dataset[target_type]))
            for ite in range(total // count):
                transfer_type = ori_type[:3] + '2' + target_type[:3]
                if start > len(vaild_dataset[ori_type]) - 1:
                    start = start % len(vaild_dataset[ori_type])
                if len(t) > 1:
                    r_img_for_fg_model = image2tensor(t2, os.path.join(val_path, ori_type),
                                                      vaild_dataset[ori_type][start:start + count], au_map,
                                                      count).cuda()
                else:
                    r_img_for_fg_model = None
                real_img = image2tensor(t1, os.path.join(val_path, ori_type),
                                        vaild_dataset[ori_type][start:start + count], au_map,
                                        count).cuda()
                desired_img = image2tensor(t1, os.path.join(val_path, target_type),
                                           vaild_dataset[target_type][start:start + count], au_map,
                                           count).cuda()
                real_au = aus_from_map(au_map, vaild_dataset[ori_type][start:start + count]).cuda()
                desired_au = aus_from_map(au_map, vaild_dataset[target_type][start:start + count]).cuda()
                if not use_step_grow:
                    vaild_transfer(model, real_img, desired_img, real_au, desired_au,
                                   os.path.join(save_path,
                                                transfer_type + '_' + str(start).zfill(3) + '.png'),
                                   save_g_only=save_g_only, model_name=model_name, fg_model=fg_model,
                                   fg_img=r_img_for_fg_model)
                else:
                    # 这个函数会生成4行，real ,desired, our_g, fg_g
                    # 第一行第二行是相同的
                    # 第三行是逐步变化
                    # 第四行是fg的变换
                    '''
                    这里输入只能count=1 total=1!!
                    '''
                    ri = real_img.repeat(len(use_step_grow), 1, 1, 1)
                    di = desired_img.repeat(len(use_step_grow), 1, 1, 1)
                    ru = real_au.repeat(len(use_step_grow), 1)
                    du = torch.zeros_like(desired_au).repeat(len(use_step_grow), 1)
                    fg_i = r_img_for_fg_model.repeat(len(use_step_grow), 1, 1, 1)
                    for s in range(len(use_step_grow)):
                        du[s][14] = use_step_grow[s]
                    vaild_transfer(model, ri, di, ru, du,
                                   os.path.join(save_path,
                                                transfer_type + '_' + str(start).zfill(3) + '.png'),
                                   save_g_only=save_g_only, model_name=model_name, fg_model=fg_model,
                                   fg_img=fg_i, use_step_grow=use_step_grow)

                save_au = desired_au.cpu().numpy()
                if save_de_au:
                    np.savetxt(
                        os.path.join(save_path, transfer_type + '_' + str(start).zfill(3) + '.txt'),
                        save_au,
                        fmt='%.3f')
                start += count
            start = 0
