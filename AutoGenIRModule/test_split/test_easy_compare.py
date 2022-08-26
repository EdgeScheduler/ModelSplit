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
import drivers
from AutoGenIRModule.pyfile.easy_model import EasyModule
from AutoGenIRModule.pyfile.easy_model_split.easy_model_0 import EasyModule_0
from AutoGenIRModule.pyfile.easy_model_split.easy_model_1 import EasyModule_1
from ModelAnalyze.easy_model_split_manual import get_whole_model_params
from ModelUtils.params_utils import parse_params_file, filter_params

input_shape = (4, 3, 14, 14)
_dtype = "float32"
input_name = "input"
mydriver = drivers.GPU()
params = get_whole_model_params()
input_data = torch.rand(4, 3, 14, 14)
input = {"input": input_data}
shape_dict = {input_name: input_data.shape}


def run_whole_model():
    ir_module = tvm.IRModule.from_expr(EasyModule())
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(ir_module, mydriver.target, params=params)
    module = graph_executor.GraphModule(lib["default"](mydriver.device))
    for k, v in input.items():
        module.set_input(k, v)
    module.run()
    output = module.get_output(0).numpy().tolist()
    print(output)
    return output


def run_split_model(params_file_path=None):
    params_dict = parse_params_file(params_file_path)
    new_input, new_params = filter_params(params_dict, 0, input, params)
    print(new_input.keys())
    print(new_params.keys())
    ir_module1 = tvm.IRModule.from_expr(EasyModule_0())
    with tvm.transform.PassContext(opt_level=3):
        lib1 = relay.build(ir_module1, mydriver.target, params=new_params)
    module1 = graph_executor.GraphModule(lib1["default"](mydriver.device))

    for k, v in new_input.items():
        module1.set_input(k, v)
    module1.run()
    output1 = module1.get_output(0).numpy().tolist()
    # print("output1=", output1)

    new_input2, new_params2 = filter_params(
        params_dict, 1, input, params, output1)
    print(new_input2.keys())
    print(new_params2.keys())
    ir_module2 = tvm.IRModule.from_expr(EasyModule_1())
    with tvm.transform.PassContext(opt_level=3):
        lib2 = relay.build(ir_module2, mydriver.target, params=new_params2)
    module2 = graph_executor.GraphModule(lib2["default"](mydriver.device))

    for k, v in new_input2.items():
        module2.set_input(k, v)
    module2.run()
    output2 = module2.get_output(0).numpy().tolist()
    print(output2)
    return output2


if __name__ == '__main__':
    whole_out = run_whole_model()
    params_file_path = "/home/onceas/wanna/ModelSplit/AutoGenIRModule/text/easy_model_split/params.json"
    split_out = run_split_model(params_file_path)
