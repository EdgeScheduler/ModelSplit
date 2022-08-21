import os
import json
import onnx
import onnxruntime
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import tvm
from config import Config
import drivers
from ModelUtils.model_utils import load_onnx_model, onnx2IRModule, build_lib, store_lib, get_lib
from tvm.contrib import graph_executor


# 模型定义
input_shape = (4, 3, 14, 14)
random_add1 = torch.rand(4, 3, 14, 14)
random_add2 = torch.rand(4, 1, 6, 6)

mydriver = drivers.GPU()
onnx_name = "easy_model"

'''
x -> conv1 ->y1
x ->add1 -> conv2 ->y2

y1+y2 -> y

y -> relu -> z1
y -> add2 -> z2

z1+z2 -> output
'''


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=1, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=3, out_channels=1, kernel_size=4, stride=2)
        self.linear = nn.Linear(in_features=4*1*6*6, out_features=1)

    def forward(self, x):                            # x: (4,3,14,14)        1, 11
        y1 = F.relu(self.conv1(x))                    # y1: (4,1,6,6)
        y2 = F.relu(self.conv2(x+random_add1))        # y2: (4,1,6,6)

        y = y1+y2

        output = F.relu(y)+y+random_add2
        return self.linear(torch.reshape(output, (1, 4*1*6*6)))

# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_ir_module():
    onnx_path = Config.ModelSavePathName(onnx_name)
    onnx_model = load_onnx_model(onnx_path)
    x = torch.rand(*input_shape, requires_grad=True)
    shape_dict = {"input": x.shape}
    mod, _ = onnx2IRModule(onnx_model, shape_dict)
    return mod


def main():
    test_count = 10
    model = Model()
    model.eval()        # 设置为推理模型

    x = torch.rand(*input_shape, requires_grad=True)
    # 计算一次前向传播，https://blog.csdn.net/qq_44930937/article/details/109701307
    _ = model(x)

    input_name = "input"
    output_name = "output"
    onnx_path = Config.ModelSavePathName(onnx_name)
    lib_path = Config.TvmLibSavePathName(
        onnx_name, mydriver.target, str(input_shape[0]))
    torch.onnx.export(model, x, Config.ModelSavePathName(onnx_name), export_params=True, input_names=[
        input_name], output_names=[output_name])  # "edge"使得自定义名称与tvm内部自动命名显示区分，便于理解

    # (N,3,224,224)——need to set input size for tvm model
    x = torch.rand(*input_shape, requires_grad=True)
    shape_dict = {input_name: x.shape}

    onnx_model = load_onnx_model(onnx_path)
    mod, params = onnx2IRModule(onnx_model, shape_dict)
    print(mod.get_global_vars())
    print(mod.get_global_type_vars())
    lib = build_lib(mod, params, mydriver.target, lib_path)
    print(type(lib))
    # print(lib.get_graph_json())
    # print(lib.function_metadata)

    if not os.path.exists(lib_path):
        store_lib(lib, lib_path)
    module = graph_executor.GraphModule(lib["default"](mydriver.device))
    print(type(module))

    # write check data to disk
    datas = {}
    for _ in range(test_count):
        data_input = torch.rand(*input_shape)
        # input-output map
        out = {}
        torch_out = model(data_input).tolist()[0]
        print("torch %s" % torch_out)
        out["pytorch"] = torch_out

        module.set_input(input_name, data_input.numpy())
        module.run()
        tvm_out = module.get_output(0).numpy().tolist()[0]
        print("tvm %s" % tvm_out)
        out["tvm"] = tvm_out
        datas[str(data_input.tolist())] = out


if __name__ == "__main__":
    main()


'''
compare torch & tvm

torch 0.6757051944732666
tvm 0.6757051944732666
torch 0.7329940795898438
tvm 0.7329941987991333
torch 0.8598699569702148
tvm 0.8598699569702148
torch 0.7072029113769531
tvm 0.7072029113769531
torch 0.742712140083313
tvm 0.742712140083313
torch 0.9915422201156616
tvm 0.9915421009063721
torch 0.9976719617843628
tvm 0.9976718425750732
torch 0.660358190536499
tvm 0.6603580713272095
torch 0.5278353691101074
tvm 0.527835488319397
torch 0.5978937149047852
tvm 0.5978938341140747
'''

'''
{
  "nodes": [
    {
      "op": "null", 
      "name": "input", 
      "inputs": []
    }, 
    {
      "op": "null", 
      "name": "p0", 
      "inputs": []
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_default_fused_nn_conv2d", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "2", 
        "out_layout": "", 
        "kernel_layout": "OIHW", 
        "hash": "13d40b07e198ac6b", 
        "func_name": "tvmgen_default_fused_nn_conv2d", 
        "data_layout": "NCHW", 
        "flatten_data": "0"
      }, 
      "inputs": [
        [
          0, 
          0, 
          0
        ], 
        [
          1, 
          0, 
          0
        ]
      ]
    }, 
    {
      "op": "null", 
      "name": "p1", 
      "inputs": []
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_default_fused_nn_bias_add", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "2", 
        "flatten_data": "0", 
        "hash": "e11d97547da10469", 
        "func_name": "tvmgen_default_fused_nn_bias_add"
      }, 
      "inputs": [
        [
          2, 
          0, 
          0
        ], 
        [
          3, 
          0, 
          0
        ]
      ]
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_default_fused_nn_relu", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "1", 
        "flatten_data": "0", 
        "hash": "688034377645acbd", 
        "func_name": "tvmgen_default_fused_nn_relu"
      }, 
      "inputs": [
        [
          4, 
          0, 
          0
        ]
      ]
    }, 
    {
      "op": "null", 
      "name": "p2", 
      "inputs": []
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_default_fused_add", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "2", 
        "flatten_data": "0", 
        "hash": "1e4d3c7b8a161fd8", 
        "func_name": "tvmgen_default_fused_add"
      }, 
      "inputs": [
        [
          0, 
          0, 
          0
        ], 
        [
          6, 
          0, 
          0
        ]
      ]
    }, 
    {
      "op": "null", 
      "name": "p3", 
      "inputs": []
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_default_fused_nn_conv2d1", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "2", 
        "data_layout": "NCHW", 
        "kernel_layout": "OIHW", 
        "hash": "13d40b07e198ac6b", 
        "func_name": "tvmgen_default_fused_nn_conv2d", 
        "out_layout": "", 
        "flatten_data": "0"
      }, 
      "inputs": [
        [
          7, 
          0, 
          0
        ], 
        [
          8, 
          0, 
          0
        ]
      ]
    }, 
    {
      "op": "null", 
      "name": "p4", 
      "inputs": []
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_default_fused_nn_bias_add1", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "2", 
        "flatten_data": "0", 
        "hash": "e11d97547da10469", 
        "func_name": "tvmgen_default_fused_nn_bias_add"
      }, 
      "inputs": [
        [
          9, 
          0, 
          0
        ], 
        [
          10, 
          0, 
          0
        ]
      ]
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_default_fused_nn_relu1", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "1", 
        "flatten_data": "0", 
        "hash": "688034377645acbd", 
        "func_name": "tvmgen_default_fused_nn_relu"
      }, 
      "inputs": [
        [
          11, 
          0, 
          0
        ]
      ]
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_default_fused_add_1", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "2", 
        "flatten_data": "0", 
        "hash": "2bc969b09eab8b94", 
        "func_name": "tvmgen_default_fused_add_1"
      }, 
      "inputs": [
        [
          5, 
          0, 
          0
        ], 
        [
          12, 
          0, 
          0
        ]
      ]
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_default_fused_nn_relu2", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "1", 
        "flatten_data": "0", 
        "hash": "688034377645acbd", 
        "func_name": "tvmgen_default_fused_nn_relu"
      }, 
      "inputs": [
        [
          13, 
          0, 
          0
        ]
      ]
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_default_fused_add_11", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "2", 
        "flatten_data": "0", 
        "hash": "2bc969b09eab8b94", 
        "func_name": "tvmgen_default_fused_add_1"
      }, 
      "inputs": [
        [
          14, 
          0, 
          0
        ], 
        [
          13, 
          0, 
          0
        ]
      ]
    }, 
    {
      "op": "null", 
      "name": "p5", 
      "inputs": []
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_default_fused_add_12", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "2", 
        "flatten_data": "0", 
        "hash": "2bc969b09eab8b94", 
        "func_name": "tvmgen_default_fused_add_1"
      }, 
      "inputs": [
        [
          15, 
          0, 
          0
        ], 
        [
          16, 
          0, 
          0
        ]
      ]
    }, 
    {
      "op": "tvm_op", 
      "name": "reshape_nop", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "1", 
        "flatten_data": "0", 
        "hash": "268ffe95a29d9a09", 
        "func_name": "__nop"
      }, 
      "inputs": [
        [
          17, 
          0, 
          0
        ]
      ]
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_default_fused_nn_batch_flatten", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "1", 
        "flatten_data": "0", 
        "hash": "38146c6a3882fb00", 
        "func_name": "tvmgen_default_fused_nn_batch_flatten"
      }, 
      "inputs": [
        [
          18, 
          0, 
          0
        ]
      ]
    }, 
    {
      "op": "null", 
      "name": "p6", 
      "inputs": []
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_default_fused_nn_dense", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "2", 
        "flatten_data": "0", 
        "hash": "3d3c2a96e0d02a43", 
        "func_name": "tvmgen_default_fused_nn_dense"
      }, 
      "inputs": [
        [
          19, 
          0, 
          0
        ], 
        [
          20, 
          0, 
          0
        ]
      ]
    }, 
    {
      "op": "null", 
      "name": "p7", 
      "inputs": []
    }, 
    {
      "op": "null", 
      "name": "p8", 
      "inputs": []
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_default_fused_multiply", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "2", 
        "flatten_data": "0", 
        "hash": "9fc35bd85efd671f", 
        "func_name": "tvmgen_default_fused_multiply"
      }, 
      "inputs": [
        [
          22, 
          0, 
          0
        ], 
        [
          23, 
          0, 
          0
        ]
      ]
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_default_fused_add_2", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "2", 
        "flatten_data": "0", 
        "hash": "11c5f8d9fd482bd0", 
        "func_name": "tvmgen_default_fused_add_2"
      }, 
      "inputs": [
        [
          21, 
          0, 
          0
        ], 
        [
          24, 
          0, 
          0
        ]
      ]
    }
  ], 
  "arg_nodes": [0, 1, 3, 6, 8, 10, 16, 20, 22, 23], 
  "heads": [
    [
      25, 
      0, 
      0
    ]
  ], 
  "attrs": {
    "dltype": [
      "list_str", 
      [
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32", 
        "float32"
      ]
    ], 
    "shape": [
      "list_shape", 
      [
        [4, 3, 14, 14], 
        [1, 3, 4, 4], 
        [4, 1, 6, 6], 
        [1], 
        [4, 1, 6, 6], 
        [4, 1, 6, 6], 
        [4, 3, 14, 14], 
        [4, 3, 14, 14], 
        [1, 3, 4, 4], 
        [4, 1, 6, 6], 
        [1], 
        [4, 1, 6, 6], 
        [4, 1, 6, 6], 
        [4, 1, 6, 6], 
        [4, 1, 6, 6], 
        [4, 1, 6, 6], 
        [4, 1, 6, 6], 
        [4, 1, 6, 6], 
        [1, 144], 
        [1, 144], 
        [1, 144], 
        [1, 1], 
        [], 
        [1], 
        [1], 
        [1, 1]
      ]
    ], 
    "device_index": [
      "list_int", 
      [
        2, 
        2, 
        2, 
        2, 
        2, 
        2, 
        2, 
        2, 
        2, 
        2, 
        2, 
        2, 
        2, 
        2, 
        2, 
        2, 
        2, 
        2, 
        2, 
        2, 
        2, 
        2, 
        2, 
        2, 
        2, 
        2
      ]
    ], 
    "storage_id": [
      "list_int", 
      [
        0, 
        1, 
        2, 
        3, 
        4, 
        2, 
        5, 
        6, 
        7, 
        4, 
        8, 
        9, 
        4, 
        9, 
        2, 
        4, 
        10, 
        2, 
        2, 
        9, 
        11, 
        12, 
        13, 
        14, 
        15, 
        16
      ]
    ]
  }, 
  "node_row_ptr": [
    0, 
    1, 
    2, 
    3, 
    4, 
    5, 
    6, 
    7, 
    8, 
    9, 
    10, 
    11, 
    12, 
    13, 
    14, 
    15, 
    16, 
    17, 
    18, 
    19, 
    20, 
    21, 
    22, 
    23, 
    24, 
    25, 
    26
  ]
}
'''
