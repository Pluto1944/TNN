from __future__ import absolute_import, division, print_function, unicode_literals
from collections import namedtuple
from torch.quantization.fake_quantize import *
import torch.nn as nn

class QConfig(namedtuple('QConfig', ['activation', 'weight'])):
    def __new__(cls, activation, weight):
        if isinstance(activation, nn.Module) or isinstance(weight, nn.Module):
            raise ValueError("QConfig received observer instance, please pass observer class instead. " +
                             "Use MyObserver.with_args(x=1) to override arguments to constructor if needed")
        return super(QConfig, cls).__new__(cls, activation, weight)


def get_default_qat_qconfig(backend='fbgemm'):
    if backend == 'fbgemm':
        return
    elif backend == 'qnnpack':
        qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                            quant_min=0,
                                                            quant_max=255,
                                                            dtype=torch.quint8,
                                                            qscheme=torch.per_tensor_symmetric,
                                                            reduce_range=False),
                          weight=default_weight_fake_quant)
    else:
        raise ValueError("Unknown backend, please specify qconfig manually")

    return qconfig