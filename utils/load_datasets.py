import random
import torch.utils.data as data
import utils.utils as util
import torch
import os

class DatasetCDNet(data.Dataset):

    def __init__(self, opt):
        super(DatasetCDNet, self).__init__()
        self.opt = opt
        self.patch_size = self.opt['P_size'] if opt['P_size'] else 64
        self.paths_GT = util.get_image_paths(opt['dataroot_GT'])
        self.paths_Noisy = util.get_image_paths(opt['dataroot_Noisy'])


    def __getitem__(self, index):

        GT_path = self.paths_GT[index]
        img_name, ext = os.path.splitext(os.path.basename(GT_path))
        img_GT = util.imread_mat(GT_path)
        Noisy_path = self.paths_Noisy[index]
        img_Noisy = util.imread_mat_y(Noisy_path)

        if self.opt['phase'] == 'train':

            noisemap_path = os.path.join('datasets/trainsets/noisemap/', '{:s}.mat'.format(img_name))
            noise_level1 = util.imread_noisemat(noisemap_path)
            noise_level = torch.FloatTensor(noise_level1)

            H, W = img_GT.shape[:2]
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_GT = img_GT[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size,:]
            patch_Noise = img_Noisy[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            mode = random.randint(0, 7)
            patch_GT = util.augment_img(patch_GT, mode=mode)
            patch_Noise = util.augment_img(patch_Noise, mode=mode)
            img_GT = util.uint2tensor3(patch_GT)
            img_Noisy = util.uint2tensor3(patch_Noise)
            noise_level_map = torch.ones((1, img_Noisy.size(1), img_Noisy.size(2))).mul_(noise_level).float()

        else:

            noisemap_path = os.path.join('datasets/testsets/Noisemap/', '{:s}.mat'.format(img_name))
            save_img_path1 = os.path.join('results/Train_results/noisy/', '{:s}.mat'.format(img_name))
            noise_level1 = util.imread_noisemat(noisemap_path)
            util.imsave_mat(img_Noisy, save_img_path1)
            noise_level = torch.FloatTensor(noise_level1)
            noise_level_map = torch.ones((1, img_Noisy.shape[0], img_Noisy.shape[1])).mul_(noise_level).float()
            img_GT, img_Noisy = util.single2tensor3(img_GT), util.single2tensor3(img_Noisy)

        img_Noisy = torch.cat((img_Noisy, noise_level_map), 0)

        return {'Noisy': img_Noisy, 'GT': img_GT, 'Noisy_path': Noisy_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)


def define_Dataset(dataset_opt):

    dataset = DatasetCDNet(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
