__all__ = [
    'DequantizeLinear'
]

import torch
from torch import nn
from torch import Tensor

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.onnx_tensor import OnnxTensor
from onnx2torch.onnx_graph import OnnxGraph

class DequantizeLinear(nn.Module, OnnxToTorchModule):
    '''
    A part of convertion QDQ onnx model to unquantized torch model.
    
    DequantizeLinear is just an operation:
    x = (x_q - zero) * scale
    
    If you want to quantize your converted model, 
    use the torch.ai.qauntization API on the generated model
    '''
    def __init__(self, scales: Tensor, zeros: Tensor, axis=1, tensor: Tensor=None, is_relu_out=False):
        super().__init__()
        self.scales = scales
        self.zeros = zeros
        self.axis = axis
        self.tensor = tensor
        self.relu = nn.ReLU()
        self.is_relu_out = is_relu_out

        if len(scales.shape) == 0:
            self.scales = scales.reshape((1))
            self.zeros = zeros.reshape((1))

        if tensor is not None:
            self.tensor = self.apply_dequantize_along_axis(tensor)

    
    def apply_dequantize_along_axis(self, x: Tensor):
        x = x.to(torch.float32)
        if len(self.scales.shape) == 0 or (len(self.scales.shape) and self.scales.shape[0] == 1):
            return (self.scales * (x-self.zeros)).requires_grad_()
        assert len(self.scales) == x.shape[self.axis]
        
        return torch.stack([
            self.scales[i] * (x_i-self.zeros[i]) for i, x_i in enumerate(torch.unbind(x, dim=self.axis))
        ], dim=self.axis).requires_grad_()
    
    
    def forward(self, input_tensor: torch.Tensor=None) -> torch.Tensor:
        if self.tensor is not None:
            return self.tensor
        if self.is_relu_out:
            return self.relu(input_tensor)
        return input_tensor


@add_converter(operation_type='DequantizeLinear', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    tensor_name = node.input_values[0]
    scale_name = node.input_values[1]
    zero_name = node.input_values[2]
 
    tensor = graph.initializers.get(tensor_name, None)
    scales = graph.initializers[scale_name].to_torch()
    zeros = graph.initializers[zero_name].to_torch()
    axis = node.attributes.get('axis', 0)

    if tensor is not None:
        tensor = tensor.to_torch()

    input_values = [tensor_name] if tensor is None else []

    # dequantize layers can be fused with activation
    # onnx adds 'Relu' at the name of such nodes (if the activation is ReLu)
    is_relu_out = 'relu' in node.name.lower()

    dql = DequantizeLinear(scales, zeros, axis, tensor, is_relu_out)
    for name in node.output_values:
        if tensor is not None and graph.initializers.get(name, None) is None: 
            graph.qinitializers[name] = OnnxTensor.from_torch(dql(), name)

    return OperationConverterResult(
        torch_module=dql,
        onnx_mapping=OnnxMapping(
            inputs=tuple(input_values),
            outputs=node.output_values,
        ),
    )
