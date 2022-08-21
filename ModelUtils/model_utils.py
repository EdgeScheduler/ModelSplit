import onnx
import tvm.relay as relay
import tvm
from config import Config
import os


def load_onnx_model(onnx_path):
    print(onnx_path)
    onnx_model = onnx.load(onnx_path)
    return onnx_model


def onnx2IRModule(onnx_model, shape_dict):
    # tvm.IRModule
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    return mod, params


def build_lib(mod, params, target, lib_path=None):
    # if (os.path.exists(lib_path)):
    # tvm.runtime.module.Module
    # return get_lib(lib_path)
    # factory_module
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    return lib


def store_lib(lib, lib_path):
    lib.export_library(lib_path)


def get_lib(lib_path):
    return tvm.runtime.load_module(lib_path)
