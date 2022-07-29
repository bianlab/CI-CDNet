import os.path
import math
import argparse
import logging
from torch.utils.data import DataLoader
import torch
import os
from utils import utils as util
from utils.load_datasets import define_Dataset
from models.model_plain import define_Model

def main(json_path='utils/train_parameter.json'):

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    opt = util.parse(parser.parse_args().opt)
    opt = util.dict_to_nonedict(opt)
    logger_name = 'train'
    util.logger_info(logger_name, os.path.join(opt['path']['root'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(util.dict2str(opt))

    model = define_Model(opt)                   #load model
    model.init_train()
    logger.info(model.info_network())
    logger.info(model.info_params())

    for phase, dataset_opt in opt['datasets'].items():      #load datasets
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            train_loader = DataLoader(train_set,
                                        batch_size=dataset_opt['dataloader_batch_size'],
                                        shuffle=dataset_opt['dataloader_shuffle'],
                                        num_workers=dataset_opt['dataloader_num_workers'],
                                        drop_last=True,
                                        pin_memory=True)

        elif phase == 'valid':
            valid_set = define_Dataset(dataset_opt)
            valid_loader = DataLoader(valid_set, batch_size=1,
                                     shuffle=True, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    current_step = 0
    for epoch in range(400):  # keep training
        for i, train_data in enumerate(train_loader):
            current_step += 1
            model.update_learning_rate(current_step)
            model.feed_data(train_data)
            model.optimize_parameters()

            #print and save results
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)
            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)
            if current_step % opt['train']['checkpoint_valid'] == 0:
                avg_psnr = 0.0
                idx = 0

                for valid_data in valid_loader:
                    idx += 1

                    model.feed_data(valid_data)
                    model.test()

                    visuals = model.current_visuals()               #save valid results
                    denoised_img = visuals['denoised']
                    GT_img = util.tensor2uint(visuals['GroundTruth'])
                    image_name_ext = os.path.basename(valid_data['Noisy_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)
                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}_amp.png'.format(img_name, current_step))
                    util.imsave(torch.abs(denoised_img), save_img_path)
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}_angle.png'.format(img_name, current_step))
                    util.imsave(torch.angle(denoised_img), save_img_path)
                    denoised_img = util.tensor2uint(denoised_img)
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.mat'.format(img_name, current_step))
                    util.imsave_mat(denoised_img, save_img_path)

                    current_psnr = util.calculate_psnr(denoised_img, GT_img)
                    avg_psnr += current_psnr
                    logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

                avg_psnr = avg_psnr / idx
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))

if __name__ == '__main__':
    main()
