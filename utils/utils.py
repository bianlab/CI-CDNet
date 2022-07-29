import os
import math
import numpy as np
import torch
import scipy.io as scio
from datetime import datetime
from torchvision.utils import save_image
import logging
from collections import OrderedDict
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.mat']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def get_image_paths(dataroot):
    paths = None  # return None if dataroot is None
    if isinstance(dataroot, str):
        paths = sorted(_get_paths_from_images(dataroot))
    elif isinstance(dataroot, list):
        paths = []
        for i in dataroot:
            paths += sorted(_get_paths_from_images(i))
    return paths


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def imread_mat(path):
    data = scio.loadmat(path)
    img = data['x']
    img = np.expand_dims(img, axis=2)
    return img

def imread_mat_y(path):
    data = scio.loadmat(path)
    img = data['y']
    img = np.expand_dims(img, axis=2)
    return img


def imread_noisemat(path):
    data = scio.loadmat(path)
    img = data['NoiseMap']
    img = img.squeeze()
    return img

def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    save_image(img, img_path, nrow=5, normalize=True)

def imsave_mat(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    scio.savemat(img_path, {'y':img})

def uint2tensor4(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0)

def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)

def tensor2uint(img):
    img = img.data.squeeze().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return img

def single2tensor3(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)

def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0)


def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def calculate_psnr(img1, img2, border=0):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    mse = np.mean(np.abs(img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

def logger_info(logger_name, log_path='default_logger.log'):

    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exist!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)

def parse(opt_path):

    json_str = ''
    with open(opt_path, 'r',encoding='utf-8') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line

    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    opt['opt_path'] = opt_path
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'dataroot_GT' in dataset and dataset['dataroot_GT'] is not None:
            dataset['dataroot_GT'] = os.path.expanduser(dataset['dataroot_GT'])
        if 'dataroot_Noisy' in dataset and dataset['dataroot_Noisy'] is not None:
            dataset['dataroot_Noisy'] = os.path.expanduser(dataset['dataroot_Noisy'])
    for key, path in opt['path'].items():
        if path and key in opt['path']:
            opt['path'][key] = os.path.expanduser(path)
    path_task = os.path.join(opt['path']['root'], opt['task'])
    opt['path']['log'] = path_task
    opt['path']['models'] = 'checkpoint'
    opt['path']['images'] = os.path.join(opt['path']['root'], 'denoised')
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    opt['num_gpu'] = len(opt['gpu_ids'])
    print('number of GPUs is: ' + str(opt['num_gpu']))

    return opt


def dict2str(opt, indent_l=1):
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None
