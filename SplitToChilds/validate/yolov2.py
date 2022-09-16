import torch
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor
from config import Config
from Onnxs.config import OnnxModelUrl
import drivers
from ModelFuntionsPython.childs.yolov2 import *
from ModelFuntionsPython.raw.yolov2 import *
from config import Config
from load_data import easy_load_from_onnx
from SplitToChilds.runtime import FilterParamsAndInput
from SplitToChilds.transfer import ModelNames

input_shape = (1, 3, 416, 416)
mydriver = drivers.GPU()
input_name = "input.1"
input_data = torch.rand(*input_shape)
input = {input_name: input_data}
shape_dict = {input_name: input_data.shape}

model_name = "yolov2"
params = {}

def run_whole_model_from_onnx():
    global params
    # (N,3,224,224)——need to set input size for tvm
    mod, params, _ =easy_load_from_onnx(model_name,shape_dict,download_url=OnnxModelUrl.Yolov2_coco,validate_download=False)
    with tvm.transform.PassContext(opt_level=0):
        lib = relay.build(mod, target=mydriver.target, params=params)
    module = graph_executor.GraphModule(lib["default"](mydriver.device))

    # tvm model
    for k, v in input.items(): 
        module.set_input(k, v)
    module.run()
    output = module.get_output(0).numpy().flatten()
    print(output[:10])


def run_whole_model():
    global params

    # print("params:", params.keys())
    ir_module = tvm.IRModule.from_expr(globals()[ModelNames[model_name]]())
    with tvm.transform.PassContext(opt_level=0):
        lib = relay.build(ir_module, mydriver.target, params=params)
    module = graph_executor.GraphModule(lib["default"](mydriver.device))
    for k, v in input.items():
        module.set_input(k, v)
    module.run()
    output = module.get_output(0).numpy().flatten()
    print(output[:10])
    return output


def run_split_model(params_dict=None):
    global params
    
    output=None
    for idx in range(len(params_dict)):
        print("--run model-%d:"%idx)
        new_input, new_params = FilterParamsAndInput(params_dict, idx, input, params,pre_output=output)
        # print("input:",list(new_input.keys()))
        # print("params:",list(new_params.keys()))
        
        ir_module = tvm.IRModule.from_expr(globals()[ModelNames[model_name]+"_"+str(idx)]())
        with tvm.transform.PassContext(opt_level=0):
            lib = relay.build(ir_module, mydriver.target, params=new_params)
        module = graph_executor.GraphModule(lib["default"](mydriver.device))

        for k, v in new_input.items():
            module.set_input(k, v)
        module.run()
        output = module.get_output(0).numpy()
    print("output=>", output.flatten()[:10])
    return output

 
if __name__ == '__main__':
    run_whole_model_from_onnx()
    run_whole_model()
    run_split_model(Config.ModelParamsFile(model_name=model_name))