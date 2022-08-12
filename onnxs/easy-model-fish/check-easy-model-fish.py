# test easy-model

from lib2to3.pgen2 import driver
import tvm
from tvm import relay
import onnx
import os
from config import Config
import drivers
from tvm.contrib import graph_executor
import json
import numpy

name="easy-model-fish"
input_info={
    "name": "input_edge",
    "shape": (4,3,14,14)
}

print("read model from: ",Config.ModelSavePathName(name))

easy_model=onnx.load(Config.ModelSavePathName(name))
mod, params = relay.frontend.from_onnx(easy_model)
# mod, params = relay.frontend.from_onnx(easy_model,{"input_edge": (4,3,14,14)})
irModule=relay.transform.InferType()(mod)                    # tvm.ir.module.IRModule

print(irModule)

# # print(type(mod))

# shape_dict = {v.name_hint : v.checked_type for v in irModule["main"].params}
# for label,type_name in shape_dict.items():
#     print(label,":",type_name)


# lib = relay.build(mod, target=drivers.GPU.target, params=params)

with relay.build_config(opt_level=0):
    graph, lib, params2 = relay.build(mod, target=drivers.CPU.target, params=params)

# with tvm.transform.PassContext(opt_level=0):
#     intrp = relay.build_module.create_executor("graph", irModule, tvm.cpu(0), target=drivers.CPU.target,params=params)

# intrp.evaluate()("...")

# with tvm.transform.PassContext(opt_level=0):
#     lib = relay.build(irModule, target=drivers.CPU.target, params={})

# graphModule = graph_executor.GraphModule(lib["default"](drivers.GPU.device))

# graphModule = graph_runtime.create(graph, lib, drivers.GPU.device)

# test_data=None
# with open(Config.ModelSaveDataPathName(name),'r') as fp:
#     test_data=json.load(fp)

# for input_test,output_test in test_data.items():
#     tvm_output = intrp.evaluate()()

#     graphModule.set_input(input_info['name'], numpy.array(list(input_test)))
#     output = graphModule.get_output(0)
#     print(output)