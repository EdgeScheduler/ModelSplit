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
import load_data

mydriver=drivers.GPU()

name="easy-model-fish"
input_info={
    "name": "input_edge",
    "shape": (4,3,14,14)
}

irModule,params,load_time= load_data.easy_load_from_onnx(name,{input_info["name"]:input_info["shape"]})
print("load model cost %ss"%(load_time))

# print(irModule)

# print(params.keys())

# shape_dict = {v.name_hint : v.checked_type for v in irModule["main"].params}
# for label,type_name in shape_dict.items():
#     print(label,":",type_name)

#############################################################################################################
# 方法一
test_data=None
flag=True
time_first=None
with tvm.transform.PassContext(opt_level=0):
    lib = relay.build(irModule, target=mydriver.target, params=params)
module = graph_executor.GraphModule(lib["default"](mydriver.device))

with open(Config.ModelSaveDataPathName(name),'r') as fp:
    test_data=json.load(fp)

start=time.time()
for input_test,output_test in test_data.items():
    module.set_input(input_info['name'], numpy.array(eval(input_test),dtype="float32"))
    
    if flag:
        time_first=time.time()
    module.run()
    if flag:
        print("first runtime: ",time.time()-time_first)
        flag=False
    tvm_output = module.get_output(0).numpy().tolist()
    if abs(tvm_output[0][0]-output_test) < 10**-4:
        continue
    else:
        print("error")
print("way-GraphModule total run time: ",time.time()-start)
#############################################################################################################

#############################################################################################################
# 方法二
start=time.time()
flag=True
time_first=None
with tvm.transform.PassContext(opt_level=0):
    intrp = relay.build_module.create_executor("graph", irModule, device=mydriver.device, target=mydriver.target,params=params)

test_data=None
with open(Config.ModelSaveDataPathName(name),'r') as fp:
    test_data=json.load(fp)

for input_test,output_test in test_data.items():
    if flag:
        time_first=time.time()
    tvm_output = intrp.evaluate()(numpy.array(eval(input_test),dtype="float32"))
    if flag:
        print("first runtime: ",time.time()-time_first)
        flag=False
    
    if abs(tvm_output.numpy().tolist()[0][0]-output_test) < 10**-4:
        continue
    else:
        print("error")
print("way-intrp run time: ",time.time()-start)
#############################################################################################################