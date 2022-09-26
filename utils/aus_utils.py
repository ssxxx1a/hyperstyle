from __future__ import print_function
from PIL import Image
import numpy as np
import os
from math import sqrt
import torchvision
from torchvision import transforms
import torch
import torch.nn.functional as F
import cv2

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_image(image_numpy, image_path):
    mkdir(os.path.dirname(image_path))
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def save_tensor_img(var, image_path, nrow=4, trans_row=3):
    """

    Args:
        var:(B,3, H, W)
        image_path:
        nrow: 每一行多少个图像
        trans_row: 有多少行需要被转换像素值

    Returns:

    """
    # var shape: (B,3, H, W)
    # var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    mkdir(os.path.dirname(image_path))

    var[:trans_row * nrow] = ((var[:trans_row * nrow] + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255

    res = torchvision.utils.make_grid(var, nrow=nrow)
    res = res.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    res = Image.fromarray(res.astype('uint8'))
    #  res=res.resize((256,256))
    res.save(image_path)


def calc_ent(x):
    """
        calculate shanno ent of x
    """
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent

def tensor2im(img, imtype=np.uint8, unnormalize=True, idx=0, nrows=None):
    if len(img.shape) == 4:
        nrows = nrows if nrows is not None else int(sqrt(img.size(0)))
        img = img[idx] if idx > 0 else torchvision.utils.make_grid(img, nrows)

    img = img.cpu().float()
    if unnormalize:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        for i, m, s in zip(img, mean, std):
            i.mul_(s).add_(m)

    image_numpy = img.detach().numpy()
    image_numpy_t = np.transpose(image_numpy, (1, 2, 0))
    image_numpy_t = image_numpy_t * 255.0

    return image_numpy_t.astype(imtype)
def save_img(img, index=0, type='mask', name='0'):
    import os
    if type == 'mask' and not os.path.exists('./analysis/mask'):
        os.makedirs('./analysis/mask')
    if type == 'sf' and not os.path.exists('./analysis/sf'):
        os.makedirs('./analysis/sf')
    from torchvision import transforms
    from PIL import Image
    import numpy as np
    from time import time
    ttt = transforms.ToPILImage()
    save_vis = torch.mean(img[index], dim=0).view(1, 128, 128).detach().cpu()
    save_vis = ttt(save_vis)
    if type == 'mask':
        save_vis.save('./analysis/mask/' + str(time()).split('.')[0] + '_' + name + '.png')
    elif type == 'sf':
        save_vis.save('./analysis/sf/' + str(time()).split('.')[0] + '_' + name + '.png')
    else:
        raise Exception('Invalid key!')

def cal_ssim_psnr_l1(transform, path1, path2, mode):
    # transform = transforms.Compose([transforms.Resize((256, 256)),
    #                           transforms.ToTensor(), ])
    from skimage.measure import compare_psnr
    from utils import cv_utils
    import pytorch_ssim
    '''

    :param transform:
    :param path1: 真实分布图list ,名字需要保持一致
    :param path2: 生成图list
    :param mode:计算什么玩意？l1 or ssim or psnr
    :return:
    '''
    vec1 = sorted(os.listdir(path1))
    vec2 = sorted(os.listdir(path2))
    res = 0.0
    for i in vec1:
        if mode != 'psnr':
            img1 = transform(Image.fromarray(cv_utils.read_cv2_img(os.path.join(path1, i)))).unsqueeze(dim=0)
            img2 = transform(Image.fromarray(cv_utils.read_cv2_img(os.path.join(path2, i)))).unsqueeze(dim=0)
            if torch.cuda.is_available():
                img1 = img1.cuda()
                img2 = img2.cuda()
        else:
            img1 = cv_utils.read_cv2_img(os.path.join(path1, i))
            img2 = cv_utils.read_cv2_img(os.path.join(path2, i))
        if mode == 'ssim':
            temp = pytorch_ssim.ssim(img1, img2)
        elif mode == 'l1':
            temp = torch.nn.functional.l1_loss(img1, img2)
        elif mode == 'psnr':
            temp = compare_psnr(img1, img2, 255)
        else:
            raise ValueError('mode error?')
        res += temp

    print('mean of all ssim score:')
    print(res / len(vec1))
def to_rgb_r(var):
    '''
    val_utils 里用到
    :param var:
    :return:
    '''
    # 3,128,128
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    res = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    res = Image.fromarray(res.astype('uint8'))
    res = res.resize((256, 256))
    return transforms.ToTensor()(res)

def tensor2maskim(mask, imtype=np.uint8, idx=0, nrows=None):
    im = tensor2im(mask, imtype=imtype, idx=idx, unnormalize=False, nrows=nrows)
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=1)
    return im
def read_cv2_img(path):
    img = cv2.imread(path, -1)

    if img is not None:
        if len(img.shape) != 3:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img




