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
from AutoGenIRModule.pyfile.resnet50 import ResnetModule
from AutoGenIRModule.pyfile.resnet50_split.resnet50_0 import ResnetModule_0
from AutoGenIRModule.pyfile.resnet50_split.resnet50_1 import ResnetModule_1
from AutoGenIRModule.pyfile.resnet50_split.resnet50_2 import ResnetModule_2
from AutoGenIRModule.pyfile.resnet50_split.resnet50_3 import ResnetModule_3
from ModelUtils.params_utils import parse_params_file, filter_params

input_shape = (1, 3, 224, 224)
mydriver = drivers.GPU()
input_name = "data"
model_name = "resnet50-v2-7"
test_count = 5
_dtype = "float32"
params = {}
input_data = torch.rand(*input_shape)
input = {input_name: input_data}
shape_dict = {input_name: input_data.shape}


def run_whole_model_from_onnx():
    global params
    # onnx model
    onnx_path = Config.ModelSavePathName(model_name)
    lib_path = Config.TvmLibSavePathByName(
        model_name, mydriver.target, str(input_shape[0]))
    # (N,3,224,224)——need to set input size for tvm

    onnx_model = load_onnx_model(onnx_path)
    mod, params = onnx2IRModule(onnx_model, shape_dict)
    fold_const = relay.transform.FoldConstant()  # 返回类型pass
    mod = fold_const(mod)
    lib = build_lib(mod, params, mydriver.target, lib_path)
    module = graph_executor.GraphModule(
        lib["default"](mydriver.device))

    # tvm model
    for k, v in input.items():
        module.set_input(k, v)
    module.run()
    output = module.get_output(0).numpy().flatten()
    print(output[:10])


def run_whole_model():
    # print("params:", params.keys())
    ir_module = tvm.IRModule.from_expr(ResnetModule())
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(ir_module, mydriver.target, params=params)
    module = graph_executor.GraphModule(lib["default"](mydriver.device))
    for k, v in input.items():
        module.set_input(k, v)
    module.run()
    output = module.get_output(0).numpy().flatten()
    print(output[:10])
    return output


def run_split_model_N(params_file_path=None, parts=1):
    # print("params:", params.keys())
    params_dict = parse_params_file(params_file_path)
    output1 = None
    for part in range(parts):
        new_input, new_params = filter_params(
            params_dict, part, input, params, output1)
        # print(new_input.keys())
        # print(new_params.keys())
        if part == 0:
            ir_module1 = tvm.IRModule.from_expr(ResnetModule_0())
        elif part == 1:
            ir_module1 = tvm.IRModule.from_expr(ResnetModule_1())
        elif part == 2:
            ir_module1 = tvm.IRModule.from_expr(ResnetModule_2())
        elif part == 3:
            ir_module1 = tvm.IRModule.from_expr(ResnetModule_3())
        with tvm.transform.PassContext(opt_level=3):
            lib1 = relay.build(ir_module1, mydriver.target, params=new_params)
        module1 = graph_executor.GraphModule(lib1["default"](mydriver.device))

        for k, v in new_input.items():
            module1.set_input(k, v)
        module1.run()
        output1 = module1.get_output(0).numpy()

    print(output1.flatten()[:10])
    return output1


if __name__ == '__main__':
    run_whole_model_from_onnx()
    run_whole_model()
    params_file_path = "/home/onceas/wanna/ModelSplit/AutoGenIRModule/text/resnet50_split/params.json"
    # run_split_model(params_file_path)
    run_split_model_N(params_file_path, 4)

'''
4 parts
[convergence_nodes[5], convergence_nodes[9], convergence_nodes[14]]
[-1.3758942   0.98943883  2.4637697   1.2364967   1.343241    0.39577398
  0.8425386  -0.9534618  -1.109506   -0.500786  ]
[-1.3758942   0.98943883  2.4637697   1.2364967   1.343241    0.39577398
  0.8425386  -0.9534618  -1.109506   -0.500786  ]
[-1.3758942   0.98943883  2.4637697   1.2364967   1.343241    0.39577398
  0.8425386  -0.9534618  -1.109506   -0.500786  ]
'''
