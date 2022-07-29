import os.path
import torch
from utils import utils as util
import time
from models.CI_CDNet import CI_CDNet as net
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor

def main():

    Amp_GT_path = 'datasets/testsets/GroundTruth_Amp'
    Pha_GT_path = 'datasets/testsets/GroundTruth_Pha'
    model_path = 'checkpoint/MixedNoise.pth'
    Denoised_path = 'results/Test_results'
    Denoised_path_amp = os.path.join(Denoised_path, 'amp')
    Denoised_path_pha = os.path.join(Denoised_path, 'pha')
    Amp_GT_paths = util.get_image_paths(Amp_GT_path)
    Pha_GT_paths = util.get_image_paths(Pha_GT_path)
    w = 512
    h = 512
    flag_speckle = True
    gauss_level_model = [20/255.0, 30/255.0, 40/255.0]          #Gaussian noise level
    speckle_level_model = [20/255.0, 30/255.0, 40/255.0]        #Speckle noise level
    noisemap_level_model = [20 / 255.0, 35 / 255.0, 45 / 255.0]     #Noisemap -- controlling denoising degree

    device = torch.device('cuda:0')
    torch.cuda.empty_cache()

    #load model
    model = net(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], downsample_mode="strideconv", upsample_mode="convtranspose", bias = False)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    sec_tim = 0
    for idx, (img_amp, img_pha) in enumerate(zip(Amp_GT_paths, Pha_GT_paths)):

        img_amp_GT = Image.open(img_amp).convert('L')
        img_amp_GT = img_amp_GT.resize((w,h), Image.BILINEAR)
        img_amp_GT = ToTensor()(img_amp_GT)
        img_amp_GT = img_amp_GT / img_amp_GT.max()

        img_pha_GT = Image.open(img_pha).convert('L')
        img_pha_GT = img_pha_GT.resize((w, h), Image.BILINEAR)
        img_pha_GT = ToTensor()(img_pha_GT)
        img_pha_GT = img_pha_GT / img_pha_GT.max()

        img_GT = img_amp_GT * np.exp(1j * img_pha_GT)

        np.random.seed(seed=0)
        noise_gaussin1 = torch.FloatTensor(np.random.normal(0,gauss_level_model[idx]/1.0,img_amp_GT.shape))
        noise_gaussin2 = torch.FloatTensor(np.random.normal(0,gauss_level_model[idx]/1.0,img_amp_GT.shape))
        img_noisy_real = img_GT.real + noise_gaussin1
        img_noisy_imag = img_GT.imag + noise_gaussin2


        if flag_speckle:
            noise_speckle1 = torch.FloatTensor(np.random.normal(0,speckle_level_model[idx]/1.0,img_amp_GT.shape))
            noise_speckle2 = torch.FloatTensor(np.random.normal(0, speckle_level_model[idx] / 1.0, img_amp_GT.shape))
            img_noisy_real = img_noisy_real + img_noisy_real * noise_speckle1
            img_noisy_imag = img_noisy_imag + img_noisy_imag * noise_speckle2

        img_Noisy = img_noisy_real + 1j * img_noisy_imag

        img_name, ext = os.path.splitext(os.path.basename(img_amp))

        noise_level_map = torch.ones((1, 1, img_Noisy.shape[1], img_Noisy.shape[2])).mul_(noisemap_level_model[idx] / 1.0).float()
        input_noise = torch.cat((img_Noisy.unsqueeze(0), noise_level_map),1)
        input_noise = input_noise.to(device)

        start_time = time.time()
        img_Denoised_real, img_Denoised_imag = model(input_noise.real, input_noise.imag)
        end_time = time.time()
        sec_tim += end_time - start_time
        img_Denoised = torch.complex(img_Denoised_real, img_Denoised_imag)

        save_img_path = os.path.join(Denoised_path_amp, '{:s}.jpg'.format(img_name))
        util.imsave(torch.abs(img_Denoised), save_img_path)
        save_img_path = os.path.join(Denoised_path_pha, '{:s}.jpg'.format(img_name))
        util.imsave(torch.angle(img_Denoised), save_img_path)

        plt.figure(img_name)
        plt.subplot(2,3,1)
        plt.imshow(torch.abs(img_Noisy).squeeze().numpy(),cmap = 'gray')
        plt.axis('off')
        plt.title('Noisy_img Amp')
        plt.subplot(2, 3, 4)
        plt.imshow(torch.angle(img_Noisy).squeeze().numpy(), cmap='gray')
        plt.axis('off')
        plt.title('Noisy_img Pha')
        plt.subplot(2, 3, 2)
        plt.imshow(torch.abs(img_Denoised).squeeze().cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.title('Denoised_img Amp')
        plt.subplot(2, 3, 5)
        plt.imshow(torch.angle(img_Denoised).squeeze().cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.title('Denoised_img Pha')
        plt.subplot(2, 3, 3)
        plt.imshow(torch.abs(img_GT).squeeze().numpy(), cmap='gray')
        plt.axis('off')
        plt.title('GroundTruth Amp')
        plt.subplot(2, 3, 6)
        plt.imshow(torch.angle(img_GT).squeeze().numpy(), cmap='gray')
        plt.axis('off')
        plt.title('GroundTruth Pha')
    plt.show()

    print('Took %s second for three noisy complex-domain images' %sec_tim)

if __name__ == '__main__':
    main()
