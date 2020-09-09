import torch
import torch.nn as nn


class AddWrap(nn.Module):

    def __init__(self, kernel_size=3, stride=2, padding=1):
        super(AddWrap, self).__init__()

        self.op = nn.quantized.FloatFunctional()

    def forward(self, x, y):
        return self.op.add(x, y)


class MulWrap(nn.Module):

    def __init__(self, kernel_size=3, stride=2, padding=1):
        super(MulWrap, self).__init__()

        self.mul_dequant = torch.quantization.DeQuantStub()
        self.mul_quant = torch.quantization.QuantStub()

    def forward(self, x, y):
        x = self.mul_dequant(x)
        y = self.mul_dequant(y)
        x = x * y
        x = self.mul_quant(x)
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
