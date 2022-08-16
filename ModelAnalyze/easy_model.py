import os
import json
import onnx
import onnxruntime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelSplit.ModelUtils.model_utils import get_tvm_model
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
    # tvm.ir.module.IRModule
    print(type(mod))
    construct_op_graph(mod)


if __name__ == '__main__':
    main()
