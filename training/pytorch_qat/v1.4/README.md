# Quantization-aware training for TRT/TNN

pytorch v1.4 quantize clone from

https://github.com/pytorch/pytorch/tree/9e7dc37f902af78b26739410766e79e63f53e27f/torch/quantization (commit:9e7dc37)



## 1. 简化版本（需手动修改模型）

**自动量化工具使用示例:**

简化版本仅有模块自动融合功能，其余功能在2. 自动版本中支持。以下以resnet为例完成resnet模型的量化，模型转换等操作。

（如果出现bug可以根据提供的resnet模型修改方案修改模型）

**一键配置**

```python
import torch
import torchvision
from quantization_wrap.quant_model import QuantizableModel

x_shape = [2, 3, 224, 224]

x = torch.ones(x_shape)

model = torchvision.models.resnet18(pretrained=True)

qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')

model_auto = QuantizableModel(model, x_shape, qconfig, black_list=[], graphoptimizer='easy')

result = model_auto(x)
```

**手动配置**

```python
import torch
import torchvision
from quantization_wrap.quant_model import QuantizableModel

x_shape = [2, 3, 224, 224]

x = torch.ones(x_shape)

model = torchvision.models.resnet18(pretrained=True)

model_auto = QuantizableModel(model, x_shape, auto=False, graphoptimizer='easy')

# Fuses modules
model_auto.fuse_model()

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights and activation

qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
model_auto.qconfig = qconfig

# black_list to add layers without quantization
black_list = []
model_auto.prepare_qat(black_list=black_list)

result = model_auto(x)
```

**模型转换**

```python
from quantization_wrap.quant_model import convert

# model convert
model_auto = convert(model_auto, inplace=True)

outputs = model_auto(x)
input_names = ["x"]

traced = torch.jit.trace(model_auto, x)

model = traced

torch.onnx.export(model, x, 'resnet_qat.onnx', input_names=input_names, example_outputs=outputs,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
```

### resnet模型修改

路径examples/models/resnet_.py下有一份修改完成的resnet模型，可以替换torchvision.models.resnet18来成功进行resnet18的量化和模型转换。

#### bug1：

##### 问题描述：

RuntimeError: Could not run 'aten::add_.Tensor' with arguments from the 'QuantizedCPUTensorId' backend. 'aten::add_.Tensor' is only available for these backends: [CPUTensorId, MkldnnCPUTensorId, SparseCPUTensorId, VariableTensorId].

##### 解决方案：

原始代码：

```python
out += identity
```

修改代码：

```python
from quantization_wrap.utils import ops

# init() 添加
self.add = ops.AddWrap()

# forword() 添加
out = self.add(out, identity)
```

#### bug2

##### 问题描述：

RuntimeError: false INTERNAL ASSERT FAILED at ..\torch\csrc\jit\passes\onnx\unpack_quantized_weights.cpp:95, please report a bug to PyTorch. Unrecognized quantized operator while trying to compute q_scale for operator aten::flatten

##### 解决方案：

原始代码：

```python
x = torch.flatten(x, 1)
```

修改代码：

```python
# init() 添加
self.flatten = ops.FlattenWrap()

# forword() 添加
x = self.flatten(x)
```

#### bug3

##### 问题描述：

RuntimeError: false INTERNAL ASSERT FAILED at ..\torch\csrc\jit\passes\onnx\unpack_quantized_weights.cpp:95, please report a bug to PyTorch. Unrecognized quantized operator while trying to compute q_scale for operator aten::max_pool2d、

##### 解决方案：

原始代码：

```python
x = self.maxpool(x)
```

修改代码：

```python
from quantization_wrap.utils import ops

# init() 修改
self.maxpool = ops.MaxPoolWrap(kernel_size=3, stride=2, padding=1)

# forword() 修改
x = self.maxpool(x)
```

## 2. 自动版本

**自动量化工具使用示例:**

目前支持resnet，mobilenetv2，shufflenet_v2，googlenet，后续模型支持持续更新。(googlenet仅支持自动量化，暂不支持ONNX转换)

**一键配置**

```python
import torch
import torchvision
from quantization_wrap.quant_model import QuantizableModel

x_shape = [2, 3, 224, 224]

x = torch.ones(x_shape)

model = torchvision.models.resnet18(pretrained=True)

qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')

model_auto = QuantizableModel(model, x_shape, qconfig, black_list=[])

result = model_auto(x)
```

**手动配置**

```python
import torch
import torchvision
from quantization_wrap.quant_model import QuantizableModel

x_shape = [2, 3, 224, 224]

x = torch.ones(x_shape)

model = torchvision.models.resnet18(pretrained=True)

model_auto = QuantizableModel(model, x_shape, auto=False)

# Fuses modules
model_auto.fuse_model()

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights and activation

qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
model_auto.qconfig = qconfig

# black_list to add layers without quantization
black_list = []
model_auto.prepare_qat(black_list=black_list)

result = model_auto(x)
```

**模型转换**

```python
from quantization_wrap.quant_model import convert

# model convert
model_auto = convert(model_auto, inplace=True)

outputs = model_auto(x)
input_names = ["x"]

traced = torch.jit.trace(model_auto, x)

model = traced

torch.onnx.export(model, x, 'resnet_qat.onnx', input_names=input_names, example_outputs=outputs,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
```

