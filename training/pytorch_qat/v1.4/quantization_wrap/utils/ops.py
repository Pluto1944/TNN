import torch
import torch.nn as nn


# class AddWrap(nn.Module):
#
#     def __init__(self):
#         super(AddWrap, self).__init__()
#
#         self.op = nn.quantized.FloatFunctional()
#
#     def forward(self, x, y):
#         return self.op.add(x, y)


class AddWrap(nn.Module):

    def __init__(self):
        super(AddWrap, self).__init__()

        self.add_dequant = torch.quantization.DeQuantStub()
        self.add_quant = torch.quantization.QuantStub()

    def forward(self, x, y):
        x = self.add_dequant(x)
        y = self.add_dequant(y)
        x = x + y
        x = self.add_quant(x)
        return x

class ReluWrap(nn.Module):

    def __init__(self):
        super(ReluWrap, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.relu_dequant = torch.quantization.DeQuantStub()
        self.relu_quant = torch.quantization.QuantStub()

    def forward(self, x):
        x = self.relu_dequant(x)
        x = self.relu(x)
        x = self.relu_quant(x)

        return x


class MulWrap(nn.Module):

    def __init__(self):
        super(MulWrap, self).__init__()

        self.mul_dequant = torch.quantization.DeQuantStub()
        self.mul_quant = torch.quantization.QuantStub()

    def forward(self, x, y):
        if type(x) == torch.Tensor:
            x = self.mul_dequant(x)
        if type(y) == torch.Tensor:
            y = self.mul_dequant(y)
        x = x * y
        x = self.mul_quant(x)
        return x


class CatWrap(nn.Module):

    def __init__(self):
        super(CatWrap, self).__init__()

        self.cat_dequant = torch.quantization.DeQuantStub()
        self.cat_quant = torch.quantization.QuantStub()

    def forward(self, x, dim=1):
        x = [self.cat_dequant(xx) for xx in x]
        x = torch.cat(x, dim=dim)
        x = self.cat_quant(x)
        return x


class ViewWrap(nn.Module):

    def __init__(self):
        super(ViewWrap, self).__init__()
        self.view_dequant = torch.quantization.DeQuantStub()
        self.view_quant = torch.quantization.QuantStub()

    def forward(self, x, shape):
        x = self.view_dequant(x)
        x = x.view(shape)
        x = self.view_quant(x)
        return x


class BatchNorm2dWrap(nn.Module):

    def __init__(self, in_channel, momentum=0.01, affine=True, eps=1.1e-5):
        super(BatchNorm2dWrap, self).__init__()
        self.bn = nn.BatchNorm2d(in_channel, momentum=momentum, affine=affine, eps=eps)

        self.dequant = torch.quantization.DeQuantStub()
        self.quant = torch.quantization.QuantStub()

    def forward(self, x):
        x = self.dequant(x)
        x = self.bn(x)
        x = self.quant(x)
        return x


class Interpolate(nn.Module):

    def __init__(self):
        super(Interpolate, self).__init__()
        self.dequant = torch.quantization.DeQuantStub()
        self.quant = torch.quantization.QuantStub()

    def forward(self, x, scale_factor=2, mode='nearest'):
        x = self.dequant(x)
        x = torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode=mode)
        x = self.quant(x)
        return x


class AdaAvgPoolWrap(nn.Module):

    def __init__(self, output_size=(1, 1)):
        super(AdaAvgPoolWrap, self).__init__()

        self.adaavgpool = nn.AdaptiveAvgPool2d(output_size)

        self.adaavgpool_dequant = torch.quantization.DeQuantStub()
        self.adaavgpool_quant = torch.quantization.QuantStub()

    def forward(self, x):
        x = self.adaavgpool_dequant(x)
        x = self.adaavgpool(x)
        x = self.adaavgpool_quant(x)
        return x


class MaxPoolWrap(nn.Module):

    def __init__(self, kernel_size=3, stride=2, padding=1):
        super(MaxPoolWrap, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

        self.maxpool_dequant = torch.quantization.DeQuantStub()
        self.maxpool_quant = torch.quantization.QuantStub()

    def forward(self, x):
        x = self.maxpool_dequant(x)
        x = self.maxpool(x)
        x = self.maxpool_quant(x)

        return x


class FlattenWrap(nn.Module):

    def __init__(self):
        super(FlattenWrap, self).__init__()

        self.flatten_dequant = torch.quantization.DeQuantStub()
        self.flatten_quant = torch.quantization.QuantStub()

    def forward(self, x):
        x = self.flatten_dequant(x)
        x = torch.flatten(x, 1)
        x = self.flatten_quant(x)
        return x
