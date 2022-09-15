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
from AutoGenIRModule.pyfile.yolov2 import YoloModule
from AutoGenIRModule.pyfile.yolov2_split.yolov2_0 import YoloModule_0
from AutoGenIRModule.pyfile.yolov2_split.yolov2_1 import YoloModule_1
from AutoGenIRModule.pyfile.yolov2_split.yolov2_2 import YoloModule_2
from AutoGenIRModule.pyfile.yolov2_split.yolov2_3 import YoloModule_3
from ModelUtils.params_utils import parse_params_file, filter_params

input_shape = (1, 3, 640, 640)
test_count = 5
mydriver = drivers.GPU()
input_name = "images"
model_name = "yolov5m6"
_dtype = "float32"
params = {}
input_data = torch.rand(*input_shape)
input = {input_name: input_data}
shape_dict = {input_name: input_data.shape}


def run_whole_model_from_onnx():
    global params
    # onnx model
    onnx_path = Config.ModelSavePathName(model_name)
    lib_path = Config.TvmLibSavePathName(
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
    ir_module = tvm.IRModule.from_expr(YoloModule())
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
            ir_module1 = tvm.IRModule.from_expr(YoloModule_0())
        elif part == 1:
            ir_module1 = tvm.IRModule.from_expr(YoloModule_1())
        elif part == 2:
            ir_module1 = tvm.IRModule.from_expr(YoloModule_2())
        elif part == 3:
            ir_module1 = tvm.IRModule.from_expr(YoloModule_3())
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
    params_file_path = "/home/onceas/wanna/ModelSplit/AutoGenIRModule/text/yolov2_split/params.json"
    # run_split_model(params_file_path)
    run_split_model_N(params_file_path, 4)

'''
2 parts
[convergence_nodes[-4]]
[-0.7175923  -0.03185541 -0.13418484 -0.3578015  -0.04517655 -0.40356982
  0.56669796 -0.64308465 -0.11933333 -0.38110426]
[-0.7175923  -0.03185541 -0.13418484 -0.3578015  -0.04517655 -0.40356982
  0.56669796 -0.64308465 -0.11933333 -0.38110426]
[-0.7175924  -0.0318551  -0.13418669 -0.35780177 -0.04517632 -0.40356833
  0.5666975  -0.6430853  -0.11933313 -0.3811031 ]

3 parts
[convergence_nodes[10], convergence_nodes[-10]]
[-0.45389566 -0.30977067 -0.16711321 -0.08704267 -0.3280568  -0.18418634
 -0.07733262 -0.28885913 -0.28817278 -0.5321783 ]
[-0.45389566 -0.30977067 -0.16711321 -0.08704267 -0.3280568  -0.18418634
 -0.07733262 -0.28885913 -0.28817278 -0.5321783 ]
[-0.45389533 -0.3097705  -0.16711459 -0.08704103 -0.32805705 -0.18418589
 -0.07733425 -0.2888586  -0.28817382 -0.532176  ]

4 parts
[convergence_nodes[10], convergence_nodes[-10], convergence_nodes[-4]]
[-0.81523955  0.29703522 -0.70980996  0.10851293  0.09290193 -0.45526552
 -0.12787952  0.30121738 -0.943975   -0.0329007 ]
[-0.81523955  0.29703522 -0.70980996  0.10851293  0.09290193 -0.45526552
 -0.12787952  0.30121738 -0.943975   -0.0329007 ]
[-0.81523883  0.29703394 -0.70981246  0.10851092  0.0929006  -0.4552661
 -0.12787901  0.30122176 -0.94397444 -0.03290022]
'''
