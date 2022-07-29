from collections import OrderedDict
import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
from models.CI_CDNet import define_net


class LLoss(nn.Module):
    def __init__(self):
        super(LLoss, self).__init__()

    def forward(self,x,y):
        loss0 = torch.mean(torch.abs(x - y))
        return loss0


class ModelBase():
    def __init__(self, opt):
        self.opt = opt                         # opt
        self.save_dir = opt['path']['models']  # save models
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.schedulers = []                   # schedulers

    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step(n)

    def current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def model_to_device(self, network):
        network = network.to(self.device)
        return network

    def describe_network(self, network):
        msg = '\n'
        msg += 'Networks name: {}'.format(network.__class__.__name__) + '\n'
        msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), network.parameters()))) + '\n'
        msg += 'Net structure:\n{}'.format(str(network)) + '\n'
        return msg

    def describe_params(self, network):
        msg = '\n'
        msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'shape', 'param_name') + '\n'
        for name, param in network.state_dict().items():
            if not 'num_batches_tracked' in name:
                v = param.data.clone().float()
                msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), v.shape, name) + '\n'
        return msg

    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)


class ModelPlain(ModelBase):
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        self.opt_train = self.opt['train']    # training option
        self.net = define_net(opt)
        self.net = self.model_to_device(self.net)


    def init_train(self):
        self.net.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log


    def save(self, iter_label):
        self.save_network(self.save_dir, self.net, 'net', iter_label)


    def define_loss(self):
        lossfn_type = self.opt_train['lossfn_type']
        if lossfn_type == 'l1':
            self.lossfn = LLoss().to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(lossfn_type))


    def define_optimizer(self):
        optim_params = []
        for k, v in self.net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.optimizer = Adam(optim_params, lr=self.opt_train['optimizer_lr'], weight_decay=0)


    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.optimizer,
                                                        self.opt_train['scheduler_milestones'],
                                                        self.opt_train['scheduler_gamma']
                                                        ))

    def feed_data(self, data, need_GT=True):
        self.Noisy = data['Noisy'].to(self.device)
        if need_GT:
            self.GT = data['GT'].to(self.device)


    def net_forward(self):
        self.Noisy_real = self.Noisy.real
        self.Noisy_imag = self.Noisy.imag
        self.Denoised_real, self.Denoised_imag = self.net(self.Noisy_real, self.Noisy_imag)
        self.Denoised = torch.complex(self.Denoised_real, self.Denoised_imag)


    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.net_forward()
        loss = self.lossfn(self.Denoised, self.GT)
        loss.backward()

        optimizer_clipgrad = self.opt_train['optimizer_clipgrad'] if self.opt_train['optimizer_clipgrad'] else 0
        if optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['optimizer_clipgrad'], norm_type=2)
        self.optimizer.step()

        self.log_dict['loss'] = loss.item()


    def test(self):
        self.net.eval()
        with torch.no_grad():
            self.net_forward()
        self.net.train()


    def current_log(self):
        return self.log_dict


    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['noisy'] = self.Noisy.detach()[0].cpu()
        out_dict['denoised'] = self.Denoised.detach()[0].cpu()
        if need_H:
            out_dict['GroundTruth'] = self.GT.detach()[0].cpu()
        return out_dict

    def print_network(self):
        msg = self.describe_network(self.net)
        print(msg)

    def print_params(self):
        msg = self.describe_params(self.net)
        print(msg)

    def info_network(self):
        msg = self.describe_network(self.net)
        return msg

    def info_params(self):
        msg = self.describe_params(self.net)
        return msg


def define_Model(opt):

    m = ModelPlain(opt)
    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m