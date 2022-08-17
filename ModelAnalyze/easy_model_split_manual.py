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
    "bias3": bias3
}
params1 = {
    "weight1": weight1,
    "weight2": weight2,
    "bias1": bias1,
    "bias2": bias2,
}
params2 = {
    "weight3": weight3,
    "bias3": bias3
}

input = {"input": input_data, "add1": random_add1, "add2": random_add2}
input1 = {"part1_input": input_data, "add1": random_add1}
input2 = {"part2_input": input_data, "add2": random_add2}

'''
%input: Tensor[(4, 3, 14, 14), float32],

%conv1.weight: Tensor[(1, 3, 4, 4), float32],
%conv1.bias: Tensor[(1), float32],
%conv2.weight: Tensor[(1, 3, 4, 4), float32],
%conv2.bias: Tensor[(1), float32],
%linear.weight: Tensor[(1, 144), float32],
%linear.bias: Tensor[(1), float32])
'''


def whole_model():
    input = relay.var("input", shape=input_shape, dtype=_dtype)
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
    return relay.Function(relay.analysis.free_vars(f), f)


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

    return relay.Function(relay.analysis.free_vars(fadd3), fadd3)


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
    return relay.Function(relay.analysis.free_vars(f), f)


def run_whole_model():
    ir_module = whole_model()
    print(ir_module)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(ir_module, mydriver.target, params=params)
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
