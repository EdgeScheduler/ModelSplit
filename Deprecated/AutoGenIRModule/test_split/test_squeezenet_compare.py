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
from AutoGenIRModule.pyfile.squeezenet1 import SqueezeNetModule
from AutoGenIRModule.pyfile.squeezenet1_split.squeezenet1_0 import SqueezeNetModule_0
from AutoGenIRModule.pyfile.squeezenet1_split.squeezenet1_1 import SqueezeNetModule_1
from AutoGenIRModule.pyfile.squeezenet1_split.squeezenet1_2 import SqueezeNetModule_2
from AutoGenIRModule.pyfile.squeezenet1_split.squeezenet1_3 import SqueezeNetModule_3
from ModelUtils.params_utils import parse_params_file, filter_params

input_shape = (1, 3, 224, 224)
mydriver = drivers.GPU()
input_name = "data_0"
model_name = "squeezenet1.0-7"
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
    ir_module = tvm.IRModule.from_expr(SqueezeNetModule())
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
        print("part=", part)
        new_input, new_params = filter_params(
            params_dict, part, input, params, output1)
        print(new_input.keys())
        # print(new_params.keys())
        if part == 0:
            ir_module1 = tvm.IRModule.from_expr(SqueezeNetModule_0())
        elif part == 1:
            ir_module1 = tvm.IRModule.from_expr(SqueezeNetModule_1())
        elif part == 2:
            ir_module1 = tvm.IRModule.from_expr(SqueezeNetModule_2())
        elif part == 3:
            ir_module1 = tvm.IRModule.from_expr(SqueezeNetModule_3())
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
    params_file_path = "/home/onceas/wanna/ModelSplit/AutoGenIRModule/text/squeezenet1_split/params.json"
    # run_split_model(params_file_path)
    run_split_model_N(params_file_path, 4)

'''
4 parts
[convergence_nodes[8], convergence_nodes[16], convergence_nodes[24]]
[3.4854227e-05 2.6267131e-03 7.8488687e-05 1.0171164e-03 9.7119075e-04
 6.9640372e-03 3.1477433e-02 1.1370956e-06 1.5933731e-05 3.0625108e-07]
[3.4854227e-05 2.6267131e-03 7.8488687e-05 1.0171164e-03 9.7119075e-04
 6.9640372e-03 3.1477433e-02 1.1370956e-06 1.5933731e-05 3.0625108e-07]
part= 0
dict_keys(['data_0'])
part= 1
dict_keys(['call_25'])
part= 2
dict_keys(['call_49'])
part= 3
dict_keys(['call_73'])
[3.4854227e-05 2.6267131e-03 7.8488687e-05 1.0171164e-03 9.7119075e-04
 6.9640372e-03 3.1477433e-02 1.1370956e-06 1.5933731e-05 3.0625108e-07]
'''
