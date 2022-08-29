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
from AutoGenIRModule.pyfile.easy_model import EasyModule


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


def gen_split_model():
    mod = get_ir_module()

    txt_name = "yolov2"
    txt_file_path = "/home/onceas/wanna/ModelSplit/AutoGenIRModule/text/{}.txt".format(
        txt_name)
    parse = MyParser(mod, txt_file_path)

    py_file_path = txt_file_path.replace("txt", "py").replace("text", "pyfile")
    parse.parse_with_text(txt_file_path)
    # module_name = txt_to_class[txt_name]
    # parse.export_py_file(module_name, py_file_path)
    parse.build_graph()
    # parse.bfs()
    convergence_nodes = parse.find_convergence_point()
    for _, node in enumerate(convergence_nodes):
        node.print_self()
    print("len=", len(convergence_nodes))
    file_path_list, params_file_path = parse.split_txt_file(
        [convergence_nodes[10], convergence_nodes[-10], convergence_nodes[-4]])
    for idx, file_path in enumerate(file_path_list):
        parse = MyParser(mod, file_path)
        py_file_path = file_path.replace(
            "txt", "py").replace("text", "pyfile")
        parse.parse_with_text(file_path)
        module_name = txt_to_class[txt_name]+"_"+str(idx)
        parse.export_py_file(module_name, py_file_path)
    return params_file_path


if __name__ == "__main__":
    gen_split_model()
