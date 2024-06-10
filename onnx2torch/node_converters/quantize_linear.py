__all__ = [
    'QuantizeLinear'
]

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.onnx_graph import OnnxGraph

class QuantizeLinear(nn.Module, OnnxToTorchModule):
    '''
    A part of convertion QDQ onnx model to unquantized torch model.
    
    In QDQ quantization type we can just skip QuantizeLinear
    
    If you want to quantize your model, 
    use the torch.ai.qauntization API on the generated model
    '''
    def __init__(self):
        super().__init__()
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor


@add_converter(operation_type='QuantizeLinear', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    tensor_name = node.input_values[0]

    input_values = [tensor_name]
    
    ql = QuantizeLinear()

    return OperationConverterResult(
        torch_module=ql,
        onnx_mapping=OnnxMapping(
            inputs=tuple(input_values),
            outputs=node.output_values,
        ),
    )
