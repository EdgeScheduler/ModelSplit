import torch
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor
from config import Config
from Onnxs.config import OnnxModelUrl
import drivers
from ModelFuntionsPython.childs.googlenet import *
from ModelFuntionsPython.raw.googlenet import *
from config import Config
from load_data import easy_load_from_onnx
from SplitToChilds.runtime import FilterParamsAndInput
from SplitToChilds.transfer import ModelNames

input_shape = (1, 3, 224, 224)
mydriver = drivers.GPU()
input_name = "data_0"
input_data = torch.rand(*input_shape)
input = {input_name: input_data}
shape_dict = {input_name: input_data.shape}

model_name = "googlenet"
params = {}

def run_whole_model_from_onnx():
    global params
    # (N,3,224,224)——need to set input size for tvm
    mod, params, _ =easy_load_from_onnx(model_name,shape_dict,download_url=OnnxModelUrl.Googlenet,validate_download=False)
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

'''
start to download file from: https://media.githubusercontent.com/media/onnx/models/main/vision/classification/inception_and_googlenet/googlenet/model/googlenet-7.onnx
success download from https://media.githubusercontent.com/media/onnx/models/main/vision/classification/inception_and_googlenet/googlenet/model/googlenet-7.onnx and save to /home/onceas/yutian/ModelSplit/Onnxs/googlenet/googlenet.onnx, cost time=522.323263s.
success to load onnx from /home/onceas/yutian/ModelSplit/Onnxs/googlenet/googlenet.onnx
One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.
[4.1861253e-04 6.0838117e-04 6.4184086e-04 8.5716334e-04 3.3068901e-03
 3.5784594e-03 1.3994859e-02 3.9392169e-05 5.0550007e-05 2.2422135e-05]
[4.16683033e-04 6.07509981e-04 6.42424973e-04 8.53189325e-04
 3.29588284e-03 3.54233989e-03 1.39002185e-02 3.94830786e-05
 5.05179378e-05 2.23895513e-05]
--run model-0:
--run model-1:
--run model-2:
--run model-3:
output=> [4.16683033e-04 6.07509981e-04 6.42424973e-04 8.53189325e-04
 3.29588284e-03 3.54233989e-03 1.39002185e-02 3.94830786e-05
 5.05179378e-05 2.23895513e-05]
'''
