# test easy-model: example for test dataset

import tvm
from tvm import relay
import onnx
import os
from config import Config
import drivers
from tvm.contrib import graph_executor
import json
import numpy
import time

name="easy-model-fish"
input_info={
    "name": "input_edge",
    "shape": (4,3,14,14)
}

print("read model from: ",Config.ModelSavePathName(name))

easy_model=onnx.load(Config.ModelSavePathName(name))
# mod, params = relay.frontend.from_onnx(easy_model)
mod, params = relay.frontend.from_onnx(easy_model,{"input_edge": (4,3,14,14)})
irModule=relay.transform.InferType()(mod)                    # tvm.ir.module.IRModule

print(irModule)

print(params.keys())

shape_dict = {v.name_hint : v.checked_type for v in irModule["main"].params}
for label,type_name in shape_dict.items():
    print(label,":",type_name)

with tvm.transform.PassContext(opt_level=0):
    lib = relay.build(mod, target=drivers.GPU.target, params=params)

module = graph_executor.GraphModule(lib["default"](drivers.GPU.device))

#############################################################################################################
# 方法一
test_data=None
with open(Config.ModelSaveDataPathName(name),'r') as fp:
    test_data=json.load(fp)

start=time.time()
for input_test,output_test in test_data.items():
    module.set_input(input_info['name'], numpy.array(eval(input_test),dtype="float32"))
    module.run()
    tvm_output = module.get_output(0).numpy().tolist()
    if abs(tvm_output[0][0]-output_test) < 10**-4:
        print("ok")
    else:
        print("error")
print("way-GraphModule run time: ",time.time()-start)
#############################################################################################################

#############################################################################################################
# 方法二
start=time.time()
with tvm.transform.PassContext(opt_level=0):
    intrp = relay.build_module.create_executor("graph", irModule, tvm.cpu(0), target=drivers.CPU.target,params=params)

test_data=None
with open(Config.ModelSaveDataPathName(name),'r') as fp:
    test_data=json.load(fp)

for input_test,output_test in test_data.items():
    tvm_output = intrp.evaluate()(numpy.array(eval(input_test),dtype="float32")).numpy().tolist()
    if abs(tvm_output[0][0]-output_test) < 10**-4:
        print("ok")
    else:
        print("error")
print("way-intrp run time: ",time.time()-start)
#############################################################################################################