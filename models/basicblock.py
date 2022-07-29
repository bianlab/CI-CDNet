from collections import OrderedDict
import torch
import torch.nn as nn

dtype = torch.float32

#complex valued block

class ComplexConv2d(nn.Module):

    def __init__(self, input_channels, output_channels,
                 kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_real = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, groups,
                                   bias)
        self.conv_imag = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, groups,
                                   bias)

    def forward(self, input_real, input_imag):
        assert input_real.shape == input_imag.shape
        return (self.conv_real(input_real.type(dtype)) - self.conv_imag(input_imag.type(dtype))), (
                    self.conv_imag(input_real.type(dtype)) + self.conv_real(input_imag.type(dtype)))


class ComplexConvTranspose2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super(ComplexConvTranspose2d, self).__init__()
        self.conv_real = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding,
                                                 output_padding, groups, bias, dilation, padding_mode)
        self.conv_imag = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding,
                                                 output_padding, groups, bias, dilation, padding_mode)

    def forward(self, input_real, input_imag):
        assert input_real.shape == input_imag.shape
        return (self.conv_real(input_real.type(dtype)) - self.conv_imag(input_imag.type(dtype))), (self.conv_imag(input_real.type(dtype)) + self.conv_real(input_imag.type(dtype)))

class ComReLU(nn.Module):
    def __init__(self):
        super(ComReLU, self).__init__()

    def forward(self, x_real, x_imag):
        return nn.ReLU(inplace=True)(x_real.type(dtype)), nn.ReLU(inplace=True)(x_imag.type(dtype))



def sequential(*args):

    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR'):
    L = []
    for t in mode:
        if t == 'C':
            L.append(ComplexConv2d(input_channels=in_channels, output_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding, bias=bias))
        elif t == 'T':
            L.append(ComplexConvTranspose2d(input_channels=in_channels, output_channels=out_channels, kernel_size=kernel_size,stride=stride, padding=padding, bias=bias))
        elif t == 'R':
            L.append(ComReLU())
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        self.c1_res = ComplexConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.ComReLU = ComReLU()
        self.c2_res = ComplexConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, input_real, input_imag):
        conv1_real, conv1_imag = self.c1_res(input_real, input_imag)
        relu_real, relu_imag = self.ComReLU(conv1_real, conv1_imag)
        conv2_real, conv2_imag = self.c2_res(relu_real, relu_imag)
        return input_real + conv2_real, input_imag + conv2_imag


def upsample_convtranspose(in_channels=64, out_channels=3, padding=0, bias=True, mode='2R'):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)
    return up1


def downsample_strideconv(in_channels=64, out_channels=64, padding=0, bias=True, mode='2R'):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)
    return down1

