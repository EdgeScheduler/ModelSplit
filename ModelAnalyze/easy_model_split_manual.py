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

input_shape = (4, 3, 14, 14)
random_add1 = torch.rand(4, 3, 14, 14)
random_add2 = torch.rand(4, 1, 6, 6)
_dtype = "float32"

input_data = torch.rand(4, 3, 14, 14)

weight1 = tvm.nd.array(np.random.uniform(-1, 1, (1, 3, 4, 4)).astype(_dtype))
weight2 = tvm.nd.array(np.random.uniform(-1, 1, (1, 3, 4, 4)).astype(_dtype))
weight3 = tvm.nd.array(np.random.uniform(-1, 1, (1, 144)).astype(_dtype))
bias1 = tvm.nd.array(np.random.uniform(-1, 1, (1,)).astype(_dtype))
bias2 = tvm.nd.array(np.random.uniform(-1, 1, (1,)).astype(_dtype))
bias3 = tvm.nd.array(np.random.uniform(-1, 1, (1,)).astype(_dtype))

mydriver = drivers.GPU()
params = {
    "weight1": weight1,
    "weight2": weight2,
    "weight3": weight3,
    "bias1": bias1,
    "bias2": bias2,
    "bias3": bias3,
    "add1": tvm.nd.array(random_add1),
    "add2": tvm.nd.array(random_add2)
}
params1 = {
    "weight1": weight1,
    "weight2": weight2,
    "bias1": bias1,
    "bias2": bias2,
    "add1": tvm.nd.array(random_add1)
}
params2 = {
    "weight3": weight3,
    "bias3": bias3,
    "add2": tvm.nd.array(random_add2)
}

input = {"input": input_data}
input1 = {"part1_input": input_data}
input2 = {"part2_input": input_data}

'''
%input: Tensor[(4, 3, 14, 14), float32],

%conv1.weight: Tensor[(1, 3, 4, 4), float32],
%conv1.bias: Tensor[(1), float32],
%conv2.weight: Tensor[(1, 3, 4, 4), float32],
%conv2.bias: Tensor[(1), float32],
%linear.weight: Tensor[(1, 144), float32],
%linear.bias: Tensor[(1), float32])
'''


def get_whole_model_params():
    return params


def whole_model():
    input = relay.var("input", shape=input_shape, dtype=_dtype)
    # add1 = relay.const(tvm.nd.array(random_add1), dtype=_dtype)
    # add2 = relay.const(tvm.nd.array(random_add2), dtype=_dtype)
    add1 = relay.var("add1", shape=random_add1.shape, dtype=_dtype)
    add2 = relay.var("add2", shape=random_add2.shape, dtype=_dtype)

    weight1 = relay.var("weight1", shape=(1, 3, 4, 4), dtype=_dtype)
    weight2 = relay.var("weight2", shape=(1, 3, 4, 4), dtype=_dtype)
    weight3 = relay.var("weight3", shape=(1, 144), dtype=_dtype)
    bias1 = relay.var("bias1", shape=(1,), dtype=_dtype)
    bias2 = relay.var("bias2", shape=(1,), dtype=_dtype)
    bias3 = relay.var("bias3", shape=(1,), dtype=_dtype)

    fadd1 = relay.add(input, add1)
    fconv1 = relay.nn.conv2d(input, weight=weight1, channels=1,
                             kernel_size=4, strides=2)
    fconv1 = relay.nn.bias_add(fconv1, bias1)
    fconv2 = relay.nn.conv2d(fadd1, weight=weight2, channels=1,
                             kernel_size=4, strides=2)
    fconv2 = relay.nn.bias_add(fconv2, bias2)
    frelu1 = relay.nn.relu(fconv1)
    frelu2 = relay.nn.relu(fconv2)
    fadd2 = relay.add(frelu1, frelu2)
    frelu3 = relay.nn.relu(fadd2)
    fadd3 = relay.add(fadd2, frelu3)

    fadd4 = relay.add(fadd3, add2)
    freshape1 = relay.reshape(fadd4, (1, 4*1*6*6))
    freshape1 = relay.nn.batch_flatten(freshape1)
    fdense1 = relay.nn.dense(data=freshape1, weight=weight3,
                             units=1, out_dtype=_dtype)
    f = relay.add(fdense1, bias3)
    return tvm.IRModule.from_expr(f)


def model_part1():
    input = relay.var("part1_input", shape=input_shape, dtype=_dtype)
    add1 = relay.var("add1", shape=random_add1.shape, dtype=_dtype)

    weight1 = relay.var("weight1", shape=(1, 3, 4, 4), dtype=_dtype)
    weight2 = relay.var("weight2", shape=(1, 3, 4, 4), dtype=_dtype)
    bias1 = relay.var("bias1", shape=(1,), dtype=_dtype)
    bias2 = relay.var("bias2", shape=(1,), dtype=_dtype)

    fadd1 = relay.add(input, add1)
    fconv1 = relay.nn.conv2d(input, weight=weight1, channels=1,
                             kernel_size=4, strides=2)
    fconv1 = relay.nn.bias_add(fconv1, bias1)
    fconv2 = relay.nn.conv2d(fadd1, weight=weight2, channels=1,
                             kernel_size=4, strides=2)
    fconv2 = relay.nn.bias_add(fconv2, bias2)
    frelu1 = relay.nn.relu(fconv1)
    frelu2 = relay.nn.relu(fconv2)
    fadd2 = relay.add(frelu1, frelu2)
    frelu3 = relay.nn.relu(fadd2)
    fadd3 = relay.add(fadd2, frelu3)
    return tvm.IRModule.from_expr(fadd3)
    # return relay.Function(relay.analysis.free_vars(fadd3), fadd3)


def model_part2():
    input2 = relay.var("part2_input", shape=random_add2.shape, dtype=_dtype)
    add2 = relay.var("add2", shape=random_add2.shape, dtype=_dtype)

    weight3 = relay.var("weight3", shape=(1, 144), dtype=_dtype)
    bias3 = relay.var("bias3", shape=(1,), dtype=_dtype)

    fadd4 = relay.add(input2, add2)
    freshape1 = relay.reshape(fadd4, (1, 4*1*6*6))
    freshape1 = relay.nn.batch_flatten(freshape1)
    fdense1 = relay.nn.dense(data=freshape1, weight=weight3,
                             units=1, out_dtype=_dtype)
    f = relay.add(fdense1, bias3)
    return tvm.IRModule.from_expr(f)


def run_whole_model():
    ir_module = whole_model()
    fold_const = relay.transform.FoldConstant()  # 返回类型pass
    ir_module = fold_const(ir_module)
    print(ir_module)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(ir_module, mydriver.target, params=params)
    # print(lib.get_graph_json())
    # print(lib.get_lib())  # Module(llvm, 4c375a8)
    module = graph_executor.GraphModule(lib["default"](mydriver.device))
    for k, v in input.items():
        module.set_input(k, v)
    module.run()
    output = module.get_output(0).numpy().tolist()
    print(output)
    return output


def run_split_model():
    ir_module1 = model_part1()
    print(ir_module1)
    with tvm.transform.PassContext(opt_level=3):
        lib1 = relay.build(ir_module1, mydriver.target, params=params1)
    module1 = graph_executor.GraphModule(lib1["default"](mydriver.device))

    for k, v in input1.items():
        module1.set_input(k, v)
    module1.run()
    output1 = module1.get_output(0).numpy().tolist()
    # print("output1=", output1)

    ir_module2 = model_part2()
    print(ir_module2)
    with tvm.transform.PassContext(opt_level=3):
        lib2 = relay.build(ir_module2, mydriver.target, params=params2)
    module2 = graph_executor.GraphModule(lib2["default"](mydriver.device))

    input2["part2_input"] = output1
    for k, v in input2.items():
        module2.set_input(k, v)
    module2.run()
    output2 = module2.get_output(0).numpy().tolist()
    print(output2)
    return output2


if __name__ == '__main__':
    whole_out = run_whole_model()
    split_out = run_split_model()
    print("whole_out:", whole_out)
    print("split_out:", split_out)


'''
fn (%input: Tensor[(4, 3, 14, 14), float32], %weight1: Tensor[(1, 3, 4, 4), float32], %bias1: Tensor[(1), float32], %add1: Tensor[(4, 3, 14, 14), float32], %weight2: Tensor[(1, 3, 4, 4), float32], %bias2: Tensor[(1), float32], %add2: Tensor[(4, 1, 6, 6), float32], %weight3: Tensor[(1, 144), float32], %bias3: Tensor[(1), float32]) {
  %0 = nn.conv2d(%input, %weight1, strides=[2, 2], padding=[0, 0, 0, 0], channels=1, kernel_size=[4, 4]);
  %1 = nn.bias_add(%0, %bias1);
  %2 = add(%input, %add1);
  %3 = nn.conv2d(%2, %weight2, strides=[2, 2], padding=[0, 0, 0, 0], channels=1, kernel_size=[4, 4]);
  %4 = nn.bias_add(%3, %bias2);
  %5 = nn.relu(%1);
  %6 = nn.relu(%4);
  %7 = add(%5, %6);
  %8 = nn.relu(%7);
  %9 = add(%7, %8);
  %10 = add(%9, %add2);
  %11 = reshape(%10, newshape=[1, 144]);
  %12 = nn.batch_flatten(%11);
  %13 = nn.dense(%12, %weight3, units=1, out_dtype="float32");
  add(%13, %bias3)
}
One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.
[[-41.173133850097656]]
fn (%part1_input: Tensor[(4, 3, 14, 14), float32], %weight1: Tensor[(1, 3, 4, 4), float32], %bias1: Tensor[(1), float32], %add1: Tensor[(4, 3, 14, 14), float32], %weight2: Tensor[(1, 3, 4, 4), float32], %bias2: Tensor[(1), float32]) {
  %0 = nn.conv2d(%part1_input, %weight1, strides=[2, 2], padding=[0, 0, 0, 0], channels=1, kernel_size=[4, 4]);
  %1 = nn.bias_add(%0, %bias1);
  %2 = add(%part1_input, %add1);
  %3 = nn.conv2d(%2, %weight2, strides=[2, 2], padding=[0, 0, 0, 0], channels=1, kernel_size=[4, 4]);
  %4 = nn.bias_add(%3, %bias2);
  %5 = nn.relu(%1);
  %6 = nn.relu(%4);
  %7 = add(%5, %6);
  %8 = nn.relu(%7);
  add(%7, %8)
}
fn (%part2_input: Tensor[(4, 1, 6, 6), float32], %add2: Tensor[(4, 1, 6, 6), float32], %weight3: Tensor[(1, 144), float32], %bias3: Tensor[(1), float32]) {
  %0 = add(%part2_input, %add2);
  %1 = reshape(%0, newshape=[1, 144]);
  %2 = nn.batch_flatten(%1);
  %3 = nn.dense(%2, %weight3, units=1, out_dtype="float32");
  add(%3, %bias3)
}
[[-41.173133850097656]]
whole_out: [[-41.173133850097656]]
split_out: [[-41.173133850097656]]
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
      "name": "add1", 
      "inputs": []
    }, 
    {
      "op": "null", 
      "name": "add2", 
      "inputs": []
    }, 
    {
      "op": "null", 
      "name": "p0", 
      "inputs": []
    }, 
    {
      "op": "null", 
      "name": "p1", 
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
          1, 
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
      "op": "null", 
      "name": "p3", 
      "inputs": []
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_default_fused_nn_conv2d_add_nn_relu", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "3", 
        "data_layout": "NCHW", 
        "kernel_layout": "OIHW", 
        "hash": "f1e84e888f3a3b2b", 
        "func_name": "tvmgen_default_fused_nn_conv2d_add_nn_relu", 
        "out_layout": "", 
        "flatten_data": "0"
      }, 
      "inputs": [
        [
          5, 
          0, 
          0
        ], 
        [
          6, 
          0, 
          0
        ], 
        [
          7, 
          0, 
          0
        ]
      ]
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_default_fused_nn_conv2d_add_nn_relu_add_nn_relu_add_add", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "5", 
        "out_layout": "", 
        "kernel_layout": "OIHW", 
        "hash": "83dd8fce37801156", 
        "func_name": "tvmgen_default_fused_nn_conv2d_add_nn_relu_add_nn_relu_add_add", 
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
          3, 
          0, 
          0
        ], 
        [
          4, 
          0, 
          0
        ], 
        [
          8, 
          0, 
          0
        ], 
        [
          2, 
          0, 
          0
        ]
      ]
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_default_fused_reshape_nn_batch_flatten", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "1", 
        "flatten_data": "0", 
        "hash": "f3abb7ecc46ce6c3", 
        "func_name": "tvmgen_default_fused_reshape_nn_batch_flatten"
      }, 
      "inputs": [
        [
          9, 
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
      "op": "null", 
      "name": "p5", 
      "inputs": []
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_default_fused_nn_dense_add", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "3", 
        "flatten_data": "0", 
        "hash": "b42046847df8cb7b", 
        "func_name": "tvmgen_default_fused_nn_dense_add"
      }, 
      "inputs": [
        [
          10, 
          0, 
          0
        ], 
        [
          11, 
          0, 
          0
        ], 
        [
          12, 
          0, 
          0
        ]
      ]
    }
  ], 
  "arg_nodes": [0, 1, 2, 3, 4, 6, 7, 11, 12], 
  "heads": [
    [
      13, 
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
        "float32"
      ]
    ], 
    "shape": [
      "list_shape", 
      [
        [4, 3, 14, 14], 
        [4, 3, 14, 14], 
        [4, 1, 6, 6], 
        [1, 3, 4, 4], 
        [1, 1, 1], 
        [4, 3, 14, 14], 
        [1, 3, 4, 4], 
        [1, 1, 1], 
        [4, 1, 6, 6], 
        [4, 1, 6, 6], 
        [1, 144], 
        [1, 144], 
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
        5, 
        6, 
        7, 
        8, 
        9, 
        8, 
        10, 
        11, 
        12
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
    14
  ]
}
'''
