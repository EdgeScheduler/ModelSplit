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
import gc
import tvm.relay as relay

input_shape = (1, 3, 224, 224)
test_count = 5
mydriver = drivers.GPU()
input_name = "data_0"


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
    gc.collect()


def main():

    # onnx model
    model_name = "squeezenet1.0-7"
    onnx_path = Config.ModelSavePathName(model_name)
    lib_path = Config.TvmLibSavePathName(
        model_name, mydriver.target, str(input_shape[0]))
    # (N,3,224,224)——need to set input size for tvm
    x = torch.rand(*input_shape, requires_grad=True)
    shape_dict = {input_name: input_shape}

    onnx_model = load_onnx_model(onnx_path)
    print(shape_dict)
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
time= 0.4627249240875244
time= 0.0036725997924804688
time= 0.003664255142211914
time= 0.0036742687225341797
time= 0.0036635398864746094
'''
