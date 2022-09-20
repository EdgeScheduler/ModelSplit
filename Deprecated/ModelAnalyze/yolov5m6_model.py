import os
import json
import onnx
import onnxruntime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelUtils.model_utils import load_onnx_model, onnx2IRModule, build_lib, store_lib, get_lib
import tvm
import onnxruntime as ort
from config import Config
import time
import drivers
from tvm.contrib import graph_executor
import tvm.relay as relay


input_shape = (1, 3, 640, 640)
test_count = 5
mydriver = drivers.GPU()
input_name = "images"
model_name = "yolov5m6"


def RunModule(lib=None):
    module = graph_executor.GraphModule(
        lib["default"](mydriver.device))

    data_input = torch.rand(*input_shape)
    # tvm model
    module.set_input(input_name, data_input.numpy())
    start = time.time()
    module.run()
    module.get_output(0).numpy().flatten()
    print("time=", time.time()-start)


def main():

    # onnx model
    onnx_path = Config.ModelSavePathName(model_name)
    lib_path = Config.TvmLibSavePathByName(
        model_name, mydriver.target, str(input_shape[0]))
    # (N,3,224,224)——need to set input size for tvm
    x = torch.rand(*input_shape, requires_grad=True)
    shape_dict = {input_name: x.shape}

    onnx_model = load_onnx_model(onnx_path)
    mod, params = onnx2IRModule(onnx_model, shape_dict)
    fold_const = relay.transform.FoldConstant()  # 返回类型pass
    mod = fold_const(mod)
    print(mod)
    lib = build_lib(mod, params, mydriver.target, lib_path)
    if not os.path.exists(lib_path):
        store_lib(lib, lib_path)

    for _ in range(test_count):
        RunModule(lib=lib)

    # print("with time_evaluator")
    # ftimer = module.module.time_evaluator(
    #     "run", mydriver.device, repeat=test_count, min_repeat_ms=500, number=1)
    # prof_res = np.array(ftimer().results)  # convert to millisecond
    # print("Mean inference time (std dev): %f s (%f s)" %
    #       (np.mean(prof_res), np.std(prof_res)))


if __name__ == "__main__":
    main()

'''
time= 2.8031797409057617
time= 0.04245591163635254
time= 0.03847146034240723
time= 0.03799319267272949
time= 0.03023052215576172
'''
