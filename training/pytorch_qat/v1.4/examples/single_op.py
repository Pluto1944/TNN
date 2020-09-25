import torch
import torch.nn as nn

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(3, 6, 5)

    def forward(self, x):
        x = self.conv(x)
        return x

class ReluNet(nn.Module):

    def __init__(self):
        super(ReluNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(x)
        return x

class AddNet(nn.Module):

    def __init__(self):
        super(AddNet, self).__init__()

        self.op = nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.op.add(x, x)

class AvgpoolNet(nn.Module):

    def __init__(self):
        super(AvgpoolNet, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.avgpool(x)
        return x

class FcNet(nn.Module):

    def __init__(self):
        super(FcNet, self).__init__()
        self.fc = nn.Linear(512, 100)

    def forward(self, x):
        x = self.fc(x)
        return x


# 更换不同的op和其对应的输入格式

x_shape = [2, 3, 224, 224]
x = torch.ones(x_shape)

model = ConvNet()
# model = ReluNet()
# model = AddNet()
# model = AvgpoolNet()

# x_shape = [2, 512]
# x = torch.ones(x_shape)
# model = FcNet()

import os, sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from quantization_wrap.quant_model import QuantizableModel
from quantization_wrap.quant_model import convert


qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')

black_list = []
model_auto = QuantizableModel(model, x_shape, qconfig, black_list=[], graphoptimizer='easy')

result = model_auto(x)

# model convert
model_auto = convert(model_auto.eval(), inplace=True)

outputs = model_auto(x)
input_names = ["x"]

traced = torch.jit.trace(model_auto, x)

model = traced

torch.onnx.export(model, x, 'single_op.onnx', input_names=input_names, example_outputs=outputs,
              operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
