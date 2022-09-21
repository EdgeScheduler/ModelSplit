import torch
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor
from config import Config
from Onnxs.download_config import OnnxModelUrl
import drivers
from config import Config
from load_data import easy_load_from_onnx
from SplitToChilds.runtime import FilterChildInput, FilterChildParams
from SplitToChilds.transfer import ModelNames
import importlib
from SplitToChilds.experience.export2lib import LoadLib
from SplitToChilds.support import SupportedModels


def RunWholeOnnxModel(model_name: str, input: dict, shape_dict: dict, driver=drivers.GPU(), onnx_download_url=OnnxModelUrl.Default, validate_download=False):
    # (N,3,224,224)——need to set input size for tvm
    mod, params, _ = easy_load_from_onnx(
        model_name, shape_dict, download_url=onnx_download_url, validate_download=validate_download)
    fold_const = relay.transform.FoldConstant()  # 返回类型pass
    mod = fold_const(mod)
    print(mod)
    with tvm.transform.PassContext(opt_level=0):
        lib = relay.build(mod, target=driver.target, params=params)
    module = graph_executor.GraphModule(lib["default"](driver.device))

    # tvm model
    for k, v in input.items():
        module.set_input(k, v)
    module.run()
    output = module.get_output(0).numpy().flatten()
    # print(output[:10])
    return output, params


def RunWholeModelByFunction(model_name: str, input: dict, params: dict, driver=drivers.GPU(), allow_lib=True):
    pythonLib = importlib.import_module(
        "ModelFuntionsPython.raw.{}".format(model_name))

    lib = None
    if allow_lib:
<<<<<<< HEAD
        lib,_ = LoadLib(model_name,driver,-1)
=======
        lib = LoadLib(model_name, driver, -1)
>>>>>>> 13d0412bde7a96006492efe49e5586d42e052bd4

    if lib is not None:
        # print("  > skip build Lib")
        pass
    else:
<<<<<<< HEAD
        if params is None:
            print("error: need params.")
            return None

        print("load function")
        ir_module = tvm.IRModule.from_expr(getattr(pythonLib,ModelNames[model_name])())
=======
        ir_module = tvm.IRModule.from_expr(
            getattr(pythonLib, ModelNames[model_name])())
>>>>>>> 13d0412bde7a96006492efe49e5586d42e052bd4
        with tvm.transform.PassContext(opt_level=0):
            lib = relay.build(ir_module, driver.target, params=params)
    module = graph_executor.GraphModule(lib["default"](driver.device))

    for k, v in input.items():
        module.set_input(k, v)
    module.run()
    output = module.get_output(0).numpy().flatten()
    # print(output[:10])
    return output
<<<<<<< HEAD
    
def RunAllChildModelSequentially(model_name:str,input:dict,params:dict=None,params_dict:dict=None, driver=drivers.GPU(),allow_lib=True):
    output=None
    for idx in range(len(params_dict)):
        # print("--run model-%d:"%idx) 
        
        output=RunChildModelByIdx(model_name,idx,input,params,params_dict,driver,output,allow_lib)
    return output

def RunChildModelByIdx(model_name:str,idx:int,input:dict,params:dict=None,params_dict:dict=None, driver=drivers.GPU(),pre_output=None,allow_lib=True):
    pythonLib=importlib.import_module("ModelFuntionsPython.childs.{}".format(model_name))
=======


def RunAllChildModelSequentially(model_name: str, input: dict, params: dict, params_dict: dict = None, driver=drivers.GPU(), allow_lib=True):
    output = None
    for idx in range(len(params_dict)):
        print("--run model-%d:" % idx)

        output = RunChildModelByIdx(
            model_name, idx, input, params, params_dict, driver, output, allow_lib)
    return output


def RunChildModelByIdx(model_name: str, idx: int, input: dict, params: dict, params_dict: dict = None, driver=drivers.GPU(), pre_output=None, allow_lib=True):
    pythonLib = importlib.import_module(
        "ModelFuntionsPython.childs.{}".format(model_name))
>>>>>>> 13d0412bde7a96006492efe49e5586d42e052bd4

    # new_input, new_params = FilterChildInput(params_dict,idx,input,pre_output),FilterChildParams(params_dict,idx,params)
    # ir_module = tvm.IRModule.from_expr(getattr(pythonLib,ModelNames[model_name]+"_"+str(idx))())
    # with tvm.transform.PassContext(opt_level=0):
    #     lib = relay.build(ir_module, driver.target, params=new_params)
    # module = graph_executor.GraphModule(lib["default"](driver.device))

    lib = None
    if allow_lib:
<<<<<<< HEAD
        lib,_ = LoadLib(model_name,driver,idx)
=======
        lib = LoadLib(model_name, driver, idx)
>>>>>>> 13d0412bde7a96006492efe49e5586d42e052bd4

    if lib is not None:
        pass
        # print("  > skip build Lib")
    else:
<<<<<<< HEAD
        if params is None:
            print("error: need params.")
            return None

        print("load function")
        new_params = FilterChildParams(params_dict,idx,params)
        ir_module = tvm.IRModule.from_expr(getattr(pythonLib,ModelNames[model_name]+"_"+str(idx))())
=======
        new_params = FilterChildParams(params_dict, idx, params)
        ir_module = tvm.IRModule.from_expr(
            getattr(pythonLib, ModelNames[model_name]+"_"+str(idx))())
>>>>>>> 13d0412bde7a96006492efe49e5586d42e052bd4
        with tvm.transform.PassContext(opt_level=0):
            lib = relay.build(ir_module, driver.target, params=new_params)
    module = graph_executor.GraphModule(lib["default"](driver.device))

    new_input = FilterChildInput(params_dict, idx, input, pre_output)
    for k, v in new_input.items():
        module.set_input(k, v)

    module.run()

    return module.get_output(0).numpy()


def RunChildModelByRange(model_name: str, start: int, end: int, input: dict, params: dict, params_dict: dict = None, driver=drivers.GPU(), pre_output=None):
    pythonLib = importlib.import_module(
        "ModelFuntionsPython.childs.{}".format(model_name))

    child_count = len(params_dict)
    if end is None or end > child_count:
        end = child_count
    elif end < 0:
        end = child_count+end

    if start < 0:
        start = child_count+start

    if not (end > start and start < child_count and start >= 0):
        print("error: bad index, out of child-range")
        return

    # print run tips
    if start == end-1:
        print("--run model (%d):" % (start))
    else:
        print("--run model (%d=>%d):" % (start, end-1))

    output = pre_output
    for idx in range(start, end):

        # print("input:",list(new_input.keys()))
        # print("params:",list(new_params.keys()))
        new_input, new_params = FilterChildInput(
            params_dict, idx, input, output), FilterChildParams(params_dict, idx, params)
        ir_module = tvm.IRModule.from_expr(
            getattr(pythonLib, ModelNames[model_name]+"_"+str(idx))())
        with tvm.transform.PassContext(opt_level=0):
            lib = relay.build(ir_module, driver.target, params=new_params)
        module = graph_executor.GraphModule(lib["default"](driver.device))

        for k, v in new_input.items():
            module.set_input(k, v)
        module.run()
        output = module.get_output(0).numpy()
    # print("output=>", output.flatten()[:10])
    return output


if __name__ == '__main__':
    model_name = "googlenet"
    model_params = SupportedModels[model_name]
    input_shape = model_params["input_shape"]
    input_name = model_params["input_name"]
    input_data = torch.rand(*input_shape)
    input = {input_name: input_data}
    shape_dict = {input_name: input_data.shape}

    output1, params = RunWholeOnnxModel(model_name, input, shape_dict)
    output2 = RunWholeModelByFunction(model_name, input, params)
    # RunAllChildModelSequentially(
    # model_name, input, params, Config.ModelParamsFile(model_name=model_name))
    print(output1[:10])
    print(output2[:10])
