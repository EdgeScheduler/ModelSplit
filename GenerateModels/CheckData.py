# test easy-model
import tvm
from tvm import relay
import onnx
import os
from config import Config
import drivers
from tvm.contrib import graph_executor
import json
import numpy

def CheckData(input_info,model_name)->bool:
    '''
    only one input is allow at this version.

    example:

    input_info={
    "name": "input_edge",
    "shape": (4,3,14,14)
    }

    '''
    onnx_path=Config.ModelSavePathName(model_name)
    if not os.path.exists(onnx_path):
        print("onnx file not exist: ",onnx_path)
    else:
        print("read model from: ",onnx_path)

    # load module
    easy_model=onnx.load(Config.ModelSavePathName(model_name))
    mod, params = relay.frontend.from_onnx(easy_model,{input_info['name']: input_info['shape']})
    irModule=relay.transform.InferType()(mod)                    # tvm.ir.module.IRModule

    with tvm.transform.PassContext(opt_level=0):
        lib = relay.build(mod, target=drivers.GPU.target, params=params)

    graphModule = graph_executor.GraphModule(lib["default"](drivers.GPU.device))

    # load data
    test_data=None
    with open(Config.ModelSaveDataPathName(model_name),'r') as fp:
        test_data=json.load(fp)

    # check same
    for input_test,output_test in test_data.items():
        if output_test<10**-6:
            print("warning: output=0.0")

        graphModule.set_input(input_info['name'], numpy.array(eval(input_test)))
        graphModule.run()
        output = graphModule.get_output(0).numpy().tolist()[0]
        if abs(output-output_test) > 10**-6:
            print("bad data: ",output,output_test)
            return False
    return True

# with tvm.transform.PassContext(opt_level=0):
#     intrp = relay.build_module.create_executor("graph", irModule, tvm.cpu(0), target=drivers.CPU.target,params=params)

# intrp.evaluate()("...")

# with tvm.transform.PassContext(opt_level=0):
#     lib = relay.build(irModule, target=drivers.CPU.target, params={})