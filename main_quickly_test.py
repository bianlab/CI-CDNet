import os.path
import torch
from utils import utils as util
import time
from models.CI_CDNet import CI_CDNet as net
import matplotlib.pyplot as plt

def main():

    Noisy_path = 'datasets/testsets/Noisy'
    GT_path = 'datasets/testsets/GroundTruth'
    Denoised_path = 'results/Test_results'
    model_path = 'checkpoint/MixedNoise.pth'
    Denoised_path_amp = os.path.join(Denoised_path, 'amp')
    Denoised_path_pha = os.path.join(Denoised_path, 'pha')
    GT_paths = util.get_image_paths(GT_path)

    noise_level_model = [36.4/255.0, 68/255.0, 71.96/255.0] # Optimal parameters for the demo data

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
    for idx, img in enumerate(GT_paths):

        img_name, ext = os.path.splitext(os.path.basename(img))
        Noisy_paths = os.path.join(Noisy_path, img_name + '.mat')
        img_Noisy = util.imread_mat_y(Noisy_paths)
        img_GT = util.imread_mat(img)
        img_Noisy = util.single2tensor4(img_Noisy)
        img_GT = util.single2tensor4(img_GT)

        noise_level_map = torch.ones((1, 1, img_Noisy.shape[2], img_Noisy.shape[3])).mul_(noise_level_model[idx] / 1.0).float()
        input_noisy = torch.cat((img_Noisy, noise_level_map),1)
        input_noisy = input_noisy.to(device)

        start_time = time.time()
        img_Denoised_real, img_Denoised_imag = model(input_noisy.real, input_noisy.imag)
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
