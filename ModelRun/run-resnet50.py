# test easy-model: example for test dataset

import tvm
from tvm import relay
import drivers
from tvm.contrib import graph_executor
import numpy
import time
import load_data

mydriver=drivers.GPU()
Count=10
name="resnet50-v2-7"
input_dict={
    "data": (1, 3, 224, 224)
}

irModule,params,load_time= load_data.easy_load_from_onnx(name,input_dict)
print("load model cost %ss"%(load_time))

with tvm.transform.PassContext(opt_level=0):
    lib = relay.build(irModule, target=mydriver.target, params=params)
module = graph_executor.GraphModule(lib["default"](mydriver.device))

for key,shape in input_dict.items():
    module.set_input(key, numpy.random.rand(*shape).astype("float32"))

for _ in range(Count):
    time_first=time.time()
    module.run()
    print("%ss"%(time.time()-time_first))