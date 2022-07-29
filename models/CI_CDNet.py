import torch
import torch.nn as nn
import models.basicblock as B
import functools
import numpy as np

#complex network structure

class CI_CDNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=1, nc=[64, 128, 256, 512], downsample_mode='strideconv', upsample_mode='convtranspose', bias=True):
        super(CI_CDNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=bias, mode='C')

        # downsample
        if downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_res1_1 = B.ResBlock(nc[0], nc[0], bias=bias)
        self.m_res1_2 = B.ResBlock(nc[0], nc[0], bias=bias)
        self.m_res1_3 = B.ResBlock(nc[0], nc[0], bias=bias)
        self.m_res1_4 = B.ResBlock(nc[0], nc[0], bias=bias)
        self.m_down1 = downsample_block(nc[0], nc[1], bias=bias, mode='2')

        self.m_res2_1 = B.ResBlock(nc[1], nc[1], bias=bias)
        self.m_res2_2 = B.ResBlock(nc[1], nc[1], bias=bias)
        self.m_res2_3 = B.ResBlock(nc[1], nc[1], bias=bias)
        self.m_res2_4 = B.ResBlock(nc[1], nc[1], bias=bias)
        self.m_down2 = downsample_block(nc[1], nc[2], bias=bias, mode='2')

        self.m_res3_1 = B.ResBlock(nc[2], nc[2], bias=bias)
        self.m_res3_2 = B.ResBlock(nc[2], nc[2], bias=bias)
        self.m_res3_3 = B.ResBlock(nc[2], nc[2], bias=bias)
        self.m_res3_4 = B.ResBlock(nc[2], nc[2], bias=bias)
        self.m_down3 = downsample_block(nc[2], nc[3], bias=bias, mode='2')

        self.m_body1 = B.ResBlock(nc[3], nc[3], bias=bias)
        self.m_body2 = B.ResBlock(nc[3], nc[3], bias=bias)
        self.m_body3 = B.ResBlock(nc[3], nc[3], bias=bias)
        self.m_body4 = B.ResBlock(nc[3], nc[3], bias=bias)

        # upsample
        if upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = upsample_block(nc[3], nc[2], bias=bias, mode='2')
        self.m_res4_1 = B.ResBlock(nc[2], nc[2], bias=bias)
        self.m_res4_2 = B.ResBlock(nc[2], nc[2], bias=bias)
        self.m_res4_3 = B.ResBlock(nc[2], nc[2], bias=bias)
        self.m_res4_4 = B.ResBlock(nc[2], nc[2], bias=bias)

        self.m_up2 = upsample_block(nc[2], nc[1], bias=bias, mode='2')
        self.m_res5_1 = B.ResBlock(nc[1], nc[1], bias=bias)
        self.m_res5_2 = B.ResBlock(nc[1], nc[1], bias=bias)
        self.m_res5_3 = B.ResBlock(nc[1], nc[1], bias=bias)
        self.m_res5_4 = B.ResBlock(nc[1], nc[1], bias=bias)

        self.m_up1 = upsample_block(nc[1], nc[0], bias=bias, mode='2')
        self.m_res6_1 = B.ResBlock(nc[0], nc[0], bias=bias)
        self.m_res6_2 = B.ResBlock(nc[0], nc[0], bias=bias)
        self.m_res6_3 = B.ResBlock(nc[0], nc[0], bias=bias)
        self.m_res6_4 = B.ResBlock(nc[0], nc[0], bias=bias)

        self.m_tail = B.conv(nc[0], out_nc, bias=bias, mode='C')

    def forward(self, x0_real, x0_imag):

        x1_real, x1_imag = self.m_head(x0_real, x0_imag)
        x2_real, x2_imag = self.m_res1_1(x1_real, x1_imag)
        x2_real, x2_imag = self.m_res1_2(x2_real, x2_imag)
        x2_real, x2_imag = self.m_res1_3(x2_real, x2_imag)
        x2_real, x2_imag = self.m_res1_4(x2_real, x2_imag)
        x2_real, x2_imag = self.m_down1(x2_real, x2_imag)

        x3_real, x3_imag = self.m_res2_1(x2_real, x2_imag)
        x3_real, x3_imag = self.m_res2_2(x3_real, x3_imag)
        x3_real, x3_imag = self.m_res2_3(x3_real, x3_imag)
        x3_real, x3_imag = self.m_res2_4(x3_real, x3_imag)
        x3_real, x3_imag = self.m_down2(x3_real, x3_imag)


        x4_real, x4_imag = self.m_res3_1(x3_real, x3_imag)
        x4_real, x4_imag = self.m_res3_2(x4_real, x4_imag)
        x4_real, x4_imag = self.m_res3_3(x4_real, x4_imag)
        x4_real, x4_imag = self.m_res3_4(x4_real, x4_imag)
        x4_real, x4_imag = self.m_down3(x4_real, x4_imag)

        x_real, x_imag = self.m_body1(x4_real, x4_imag)
        x_real, x_imag = self.m_body2(x_real, x_imag)
        x_real, x_imag = self.m_body3(x_real, x_imag)
        x_real, x_imag = self.m_body4(x_real, x_imag)

        x_real, x_imag = self.m_up3(x_real+x4_real, x_imag+x4_imag)
        x_real, x_imag = self.m_res4_1(x_real, x_imag)
        x_real, x_imag = self.m_res4_2(x_real, x_imag)
        x_real, x_imag = self.m_res4_3(x_real, x_imag)
        x_real, x_imag = self.m_res4_4(x_real, x_imag)

        x_real, x_imag = self.m_up2(x_real+x3_real, x_imag+x3_imag)
        x_real, x_imag = self.m_res5_1(x_real, x_imag)
        x_real, x_imag = self.m_res5_2(x_real, x_imag)
        x_real, x_imag = self.m_res5_3(x_real, x_imag)
        x_real, x_imag = self.m_res5_4(x_real, x_imag)

        x_real, x_imag = self.m_up1(x_real+x2_real, x_imag+x2_imag)
        x_real, x_imag = self.m_res6_1(x_real, x_imag)
        x_real, x_imag = self.m_res6_2(x_real, x_imag)
        x_real, x_imag = self.m_res6_3(x_real, x_imag)
        x_real, x_imag = self.m_res6_4(x_real, x_imag)

        x_real, x_imag = self.m_tail(x_real+x1_real, x_imag+x1_imag)

        return x_real, x_imag



def define_net(opt):
    opt_net = opt['net']

    net = CI_CDNet(in_nc=opt_net['in_nc'],
                out_nc=opt_net['out_nc'],
                nc=opt_net['nc'],
                downsample_mode=opt_net['downsample_mode'],
                upsample_mode=opt_net['upsample_mode'],
                bias=opt_net['bias'])

    init_weights(net,
                    init_type=opt_net['init_type'],
                    init_bn_type=opt_net['init_bn_type'],
                    gain=opt_net['init_gain'])

    return net

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")
    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()  # tensor.size(3)tensor.size(4)*....
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out

def complex_kaiming_normal_(tensor_real, mode='fan_in'):

    fan = _calculate_correct_fan(tensor_real, mode)
    s = 1. / fan
    rng = np.random.RandomState()
    modulus = rng.rayleigh(scale=s, size=tensor_real.shape)
    phase = rng.uniform(low=-np.pi, high=np.pi, size=tensor_real.shape)
    weight_real = modulus * np.cos(phase)
    weight_imag = modulus * np.sin(phase)

    with torch.no_grad():
        return torch.FloatTensor(weight_real), torch.FloatTensor(weight_imag)


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('ComplexConv') != -1 or classname.find('Linear') != -1:
            if init_type == 'kaiming':
                m_real, m_imag = complex_kaiming_normal_(m.conv_real.weight)
                m.conv_real.weight = torch.nn.Parameter(m_real)
                m.conv_imag.weight = torch.nn.Parameter(m_imag)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))
        elif classname.find('ResBlock') != -1:
            m_real, m_imag = complex_kaiming_normal_(m.c1_res.conv_real.weight)
            m.c1_res.conv_real.weight = torch.nn.Parameter(m_real)
            m.c1_res.conv_real.weight = torch.nn.Parameter(m_imag)
            m_real, m_imag = complex_kaiming_normal_(m.c2_res.conv_real.weight)
            m.c2_res.conv_real.weight = torch.nn.Parameter(m_real)
            m.c2_res.conv_real.weight = torch.nn.Parameter(m_imag)

    if init_type not in ['default', 'none']:
        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network defination!')


