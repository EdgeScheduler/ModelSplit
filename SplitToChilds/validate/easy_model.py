import torch
import tvm
import numpy as np
import tvm.relay as relay
from tvm.contrib import graph_executor
import drivers
from ModelFuntionsPython.childs.easy_model import *
from config import Config
from ModelFuntionsPython.raw.easy_model import *
from SplitToChilds.runtime import FilterParamsAndInput
from SplitToChilds.transfer import ModelNames

model_name="easy_model"

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

input_shape = (4, 3, 14, 14)
_dtype = "float32"
input_name = "input"
mydriver = drivers.GPU()
input_data = torch.rand(4, 3, 14, 14)
input = {"input": input_data}
shape_dict = {input_name: input_data.shape}


def run_whole_model():
    ir_module = tvm.IRModule.from_expr(globals()[ModelNames[model_name]]())
    with tvm.transform.PassContext(opt_level=0):
        lib = relay.build(ir_module, mydriver.target, params=params)
    module = graph_executor.GraphModule(lib["default"](mydriver.device))
    for k, v in input.items():
        module.set_input(k, v)
    module.run()
    output = module.get_output(0).numpy().tolist()
    print(output)
    return output


def run_split_model(params_dict=None):
    output=None
    for idx in range(len(params_dict)):
        print("--run model-%d:"%idx)
        new_input, new_params = FilterParamsAndInput(params_dict, idx, input, params,pre_output=output)
        print("input:",list(new_input.keys()))
        print("params:",list(new_params.keys()))
        
        ir_module = tvm.IRModule.from_expr(globals()[ModelNames[model_name]+"_"+str(idx)]())
        with tvm.transform.PassContext(opt_level=0):
            lib = relay.build(ir_module, mydriver.target, params=new_params)
        module = graph_executor.GraphModule(lib["default"](mydriver.device))

        for k, v in new_input.items():
            module.set_input(k, v)
        module.run()
        output = module.get_output(0).numpy().tolist()
    print("output=>", output)


if __name__ == '__main__':
    whole_out = run_whole_model()
    split_out = run_split_model(Config.ModelParamsFile(model_name=model_name))
