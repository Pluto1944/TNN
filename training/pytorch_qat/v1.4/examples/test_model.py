import torch
import torch.nn as nn

x_shape = [2, 3, 224, 224]
x = torch.ones(x_shape)

from models.resnet_ import resnet18
model = resnet18(pretrained=True)
result_fp32 = model(x)
print("******** fp32 model ********")
#print(model)

############# step1. add fake op in fp32 model
print("step1. add fake op ********")
import os, sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from quantization_wrap.quant_model import QuantizableModel
from quantization_wrap.quant_model import convert
from quantization_wrap.quantization.symmetric_qconfig import get_default_qat_qconfig

#qconfig = get_default_qat_qconfig('qnnpack')
qconfig = get_default_qat_qconfig('qnnpack_perchannel')

black_list = []
model_auto = QuantizableModel(model, x_shape, qconfig, black_list=[], graphoptimizer='easy')
result = model_auto(x)

# wrap_optimizer
from quantization_wrap.graph.graph_optimizer import wrap_optimizer
wrap_optimizer(model_auto, x)

print("******** fake model ********")
#print(model_auto)
#model training!

############# step2. convert pytorch fake model to pytorch int8 model
print("step2. convert to int8 model ********")
model_auto = convert(model_auto.eval(), inplace=True)
print("******** int8 model ********")
#print(model_auto)


############# step3. convert pytorch int8 model to onnx int8 model
print("step2. convert to onnx model ********")
outputs = model_auto(x)
input_names = ["x"]

traced = torch.jit.trace(model_auto, x)

model = traced

torch.onnx.export(model, x, 'test_model.onnx', input_names=input_names, example_outputs=outputs,
              operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
