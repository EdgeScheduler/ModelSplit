import os
import json
import onnx
import onnxruntime
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import tvm_model
import tvm

onnx_name = "easy-model"
onnx_fold = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../onnxs", onnx_name)
os.makedirs(onnx_fold, exist_ok=True)

onnx_path = os.path.join(onnx_fold, onnx_name+".onnx")
input_name = "input"
output_name = "output"

# 模型定义
input_shape = (4, 3, 14, 14)

if __name__ == '__main__':
    x = torch.rand(*input_shape, requires_grad=True)
    shape_dict = {input_name: x.detach().numpy().shape}
    module, params = tvm_model.get_tvm_model(
        onnx_path, {}, target="cuda", dev=tvm.cuda())
