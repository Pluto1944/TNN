import torch
import torch.nn as nn


class AddWrap(nn.Module):

    def __init__(self, kernel_size=3, stride=2, padding=1):
        super(AddWrap, self).__init__()

        self.op = nn.quantized.FloatFunctional()

    def forward(self, x, y):
        return self.op.add(x, y)


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
