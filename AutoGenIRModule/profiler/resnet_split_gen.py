from GenerateModels.easy_model import get_ir_module
from AutoGenIRModule.gen_irmodule import MyParser
import numpy as np
from ModelUtils.model_utils import load_onnx_model, onnx2IRModule, build_lib, store_lib, get_lib
import tvm.relay as relay
from tvm.contrib import graph_executor
import onnxruntime as ort
from config import Config
from relayIR.relay_graph import construct_op_graph
import drivers
import torch
import torch.nn as nn
import torch.nn.functional as F
import tvm
from relayIR.relay_graph import construct_op_graph
from AutoGenIRModule.profiler.pyfile.resnet50_split.resnet50_0 import ResnetModule_0
from AutoGenIRModule.profiler.pyfile.resnet50_split.resnet50_1 import ResnetModule_1
from ModelUtils.params_utils import parse_params_file, filter_params
import sys

txt_to_class = {
    "googlenet": "GoogleNetModule",
    "resnet50": "ResnetModule",
    "easy_model": "EasyModule",
    "yolov2": "YoloModule",
    "squeezenet1": "SqueezeNetModule",
    "mobilenetv2": "MobileNetModule",
    "mobilenetv2_back": "MobileNetModule",
    "vgg19": "Vgg19Module",
}

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


def get_params():
    global params
    # onnx model
    onnx_path = Config.ModelSavePathName(model_name)
    lib_path = Config.TvmLibSavePathName(
        model_name, mydriver.target, str(input_shape[0]))
    # (N,3,224,224)——need to set input size for tvm

    onnx_model = load_onnx_model(onnx_path)
    _, params = onnx2IRModule(onnx_model, shape_dict)


def split_model_profile(params_file_path=None, parts=1):
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
        with tvm.transform.PassContext(opt_level=3):
            lib1 = relay.build(ir_module1, mydriver.target, params=new_params)
        module1 = graph_executor.GraphModule(lib1["default"](mydriver.device))

        for k, v in new_input.items():
            module1.set_input(k, v)
        module1.run()
        output1 = module1.get_output(0).numpy()

    print(output1.flatten()[:10])
    return output1


def gen_split_model(index):
    txt_name = "resnet50"
    txt_file_path = "/home/onceas/wanna/ModelSplit/AutoGenIRModule/profiler/text/{}.txt".format(
        txt_name)
    parse = MyParser(None, txt_file_path)

    # py_file_path = txt_file_path.replace("txt", "py").replace("text", "pyfile")
    parse.parse_with_text(txt_file_path)
    # module_name = txt_to_class[txt_name]
    # parse.export_py_file(module_name, py_file_path)
    parse.build_graph()
    # parse.bfs()
    convergence_nodes = parse.find_convergence_point()
    # for _, node in enumerate(convergence_nodes):
    # node.print_self()
    print("len=", len(convergence_nodes))

    for nidx, node in enumerate([convergence_nodes[index]]):
        print("nidx={} opidx={}".format(nidx, node.layer.name))
        file_path_list, params_file_path = parse.split_txt_file(
            [convergence_nodes[nidx]])
        for idx, file_path in enumerate(file_path_list):
            split_parse = MyParser(None, file_path)

            split_parse.parse_with_text(file_path)
            module_name = txt_to_class[txt_name]+"_"+str(idx)

            # py_file_path = file_path.replace(
            # "txt", "py").replace("text", "pyfile/{}_{}".format(txt_name, nidx))
            py_file_path = file_path.replace(
                "txt", "py").replace("text", "pyfile")
            split_parse.export_py_file(module_name, py_file_path)

        # profile
        params_file_path = "/home/onceas/wanna/ModelSplit/AutoGenIRModule/profiler/text/resnet50_split/params.json"
        # run_split_model(params_file_path)
        split_model_profile(params_file_path, 2)

    return params_file_path


if __name__ == "__main__":
    index = int(sys.argv[1])
    get_params()
    gen_split_model(index)

'''
arg=1-------------------------22-08-29-08:39:19 AM-UTC
max_used: 546.4375, max_util: 0
arg=2-------------------------22-08-29-08:40:13 AM-UTC
max_used: 542.4375, max_util: 0
arg=3-------------------------22-08-29-08:41:08 AM-UTC
max_used: 542.4375, max_util: 0
arg=4-------------------------22-08-29-08:42:02 AM-UTC
max_used: 542.4375, max_util: 0
arg=5-------------------------22-08-29-08:42:56 AM-UTC
max_used: 542.4375, max_util: 0
arg=6-------------------------22-08-29-08:43:51 AM-UTC
max_used: 542.4375, max_util: 0
arg=7-------------------------22-08-29-08:44:45 AM-UTC
max_used: 546.4375, max_util: 0
arg=8-------------------------22-08-29-08:45:39 AM-UTC
max_used: 546.4375, max_util: 0
arg=9-------------------------22-08-29-08:46:34 AM-UTC
max_used: 542.4375, max_util: 0
arg=10-------------------------22-08-29-08:47:28 AM-UTC
max_used: 542.4375, max_util: 0
arg=11-------------------------22-08-29-08:48:22 AM-UTC
max_used: 542.4375, max_util: 0
arg=12-------------------------22-08-29-08:49:17 AM-UTC
max_used: 542.4375, max_util: 0
arg=13-------------------------22-08-29-08:50:11 AM-UTC
max_used: 542.4375, max_util: 0
arg=14-------------------------22-08-29-08:51:05 AM-UTC
max_used: 542.4375, max_util: 0
arg=15-------------------------22-08-29-08:52:00 AM-UTC
max_used: 542.4375, max_util: 0
'''
