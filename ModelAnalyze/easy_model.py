import os
import json
import onnx
import onnxruntime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelUtils.model_utils import load_onnx_model, onnx2IRModule, build_lib, store_lib, get_lib
import load_data
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor
import onnxruntime as ort
from config import Config
import time
import drivers
from relayIR.relay_graph import construct_op_graph

input_shape = (4, 3, 14, 14)
input_name = "input"
test_count = 5
mydriver = drivers.GPU()


def main():
    # onnx model
    onnx_path = Config.ModelSavePathName("easy_model")
    x = torch.rand(*input_shape, requires_grad=True)
    shape_dict = {input_name: x.shape}
    # module, _ = get_tvm_model(
    #     onnx_path, shape_dict, target=mydriver.target, dev=mydriver.device)

    onnx_model = onnx.load(onnx_path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    func = mod.functions.items()[0][1]
    # print(func)

    print(mod.astext(show_meta_data=True))
    print(mod)
    print(type(mod))
    print(func.params)
    print(type(func.params))
    print(type(func))

    # tvm.ir.module.IRModule
    # construct_op_graph(mod)


if __name__ == '__main__':
    main()


'''
def @main(%input: Tensor[(4, 3, 14, 14), float32], %conv1.weight: Tensor[(1, 3, 4, 4), float32], %conv1.bias: Tensor[(1), float32], %conv2.weight: Tensor[(1, 3, 4, 4), float32], %conv2.bias: Tensor[(1), float32], %linear.weight: Tensor[(1, 144), float32], %linear.bias: Tensor[(1), float32]) {
  %0 = nn.conv2d(%input, %conv1.weight, strides=[2, 2], padding=[0, 0, 0, 0], channels=1, kernel_size=[4, 4]);
  %1 = nn.bias_add(%0, %conv1.bias);
  %2 = add(%input, meta[relay.Constant][0]);
  %3 = nn.conv2d(%2, %conv2.weight, strides=[2, 2], padding=[0, 0, 0, 0], channels=1, kernel_size=[4, 4]);
  %4 = nn.bias_add(%3, %conv2.bias);
  %5 = nn.relu(%1);
  %6 = nn.relu(%4);
  %7 = add(%5, %6);
  %8 = nn.relu(%7);
  %9 = add(%8, %7);
  %10 = add(%9, meta[relay.Constant][1]);
  %11 = reshape(%10, newshape=[1, 144]);
  %12 = nn.batch_flatten(%11);
  %13 = nn.dense(%12, %linear.weight, units=1);
  %14 = multiply(1f, %linear.bias);
  add(%13, %14)
}
'''
