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

input_shape = (1, 3, 640, 640)
test_count = 5
mydriver = drivers.GPU()
input_name = "images"
model_name = "yolov5m6"
_dtype = "float32"
params = {}
x = torch.rand(*input_shape, requires_grad=True)
input = {input_name: x.detach().numpy()}

# os.environ['TVM_BACKTRACE'] = "1"


def run_init_model():
    global params
    onnx_path = Config.ModelSavePathName(model_name)
    lib_path = Config.TvmLibSavePathByName(
        model_name, mydriver.target, str(input_shape[0]))
    # (N,3,224,224)——need to set input size for tvm
    shape_dict = {input_name: x.shape}

    onnx_model = load_onnx_model(onnx_path)
    mod, params = onnx2IRModule(onnx_model, shape_dict)
    with open("./AutoGenIRModule/params/"+model_name+".txt", "w", encoding="utf-8") as fp:
        fp.write(str(params))
    # lib = build_lib(mod, params, mydriver.target, lib_path)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=mydriver.target, params=params)
    module = graph_executor.GraphModule(lib["default"](mydriver.device))
    for k, v in input.items():
        module.set_input(k, v)
    module.run()
    output = module.get_output(0).numpy()
    output = output.flatten()[:10]
    print(output)
    return output


def run_rebuild_model():
    global params
    ir_module = tvm.IRModule.from_expr(YoloModule())
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(ir_module, target=mydriver.target, params=params)
    module = graph_executor.GraphModule(lib["default"](mydriver.device))
    for k, v in input.items():
        module.set_input(k, v)
    module.run()
    output = module.get_output(0).numpy()
    output = output.flatten()[:10]
    print(output)
    return output


if __name__ == '__main__':
    init_out = run_init_model()
    rebuild_out = run_rebuild_model()


'''
[-0.7798415   0.08921957 -0.20452632 -0.16593786  0.22566657 -0.4724225
  0.0481599  -0.09339008 -0.81382245 -0.09246691]
[-0.7798415   0.08921957 -0.20452632 -0.16593786  0.22566657 -0.4724225
  0.0481599  -0.09339008 -0.81382245 -0.09246691]
'''
