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
from AutoGenIRModule.pyfile.googlenet import GoogleNetModule
from AutoGenIRModule.pyfile.googlene_split.googlene_0 import GoogleNetModule_0
from AutoGenIRModule.pyfile.googlene_split.googlene_1 import GoogleNetModule_1
from AutoGenIRModule.pyfile.googlene_split.googlene_2 import GoogleNetModule_2
from AutoGenIRModule.pyfile.googlene_split.googlene_3 import GoogleNetModule_3
from ModelUtils.params_utils import parse_params_file, filter_params

input_shape = (1, 3, 224, 224)
mydriver = drivers.GPU()
test_count = 5
input_name = "data_0"
model_name = "googlenet-7"
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
    module = graph_executor.GraphModule(lib["default"](mydriver.device))

    # tvm model
    for k, v in input.items():
        module.set_input(k, v)
    module.run()
    output = module.get_output(0).numpy().flatten()
    print(output[:10])


def run_whole_model():
    # print("params:", params.keys())
    ir_module = tvm.IRModule.from_expr(GoogleNetModule())
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
            ir_module1 = tvm.IRModule.from_expr(GoogleNetModule_0())
        elif part == 1:
            ir_module1 = tvm.IRModule.from_expr(GoogleNetModule_1())
        elif part == 2:
            ir_module1 = tvm.IRModule.from_expr(GoogleNetModule_2())
        elif part == 3:
            ir_module1 = tvm.IRModule.from_expr(GoogleNetModule_3())
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
    params_file_path = "/home/onceas/wanna/ModelSplit/AutoGenIRModule/text/googlene_split/params.json"
    # run_split_model(params_file_path)
    run_split_model_N(params_file_path, 4)

'''
4 parts
[convergence_nodes[5], convergence_nodes[9], convergence_nodes[14]]
[4.5174066e-04 6.4512115e-04 6.6901097e-04 9.1515202e-04 3.5159728e-03
 3.7337572e-03 1.4605894e-02 4.1007763e-05 5.2538850e-05 2.3463170e-05]
[4.4942874e-04 6.4444938e-04 6.6888583e-04 9.0988656e-04 3.5026169e-03
 3.6950856e-03 1.4501633e-02 4.1110681e-05 5.2522253e-05 2.3423481e-05]
[4.4942874e-04 6.4444938e-04 6.6888583e-04 9.0988656e-04 3.5026169e-03
 3.6950856e-03 1.4501633e-02 4.1110681e-05 5.2522253e-05 2.3423481e-05]
'''
