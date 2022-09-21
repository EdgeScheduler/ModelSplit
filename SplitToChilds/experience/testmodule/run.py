import torch
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor
from config import Config
from Onnxs.download_config import OnnxModelUrl
import drivers
from config import Config
from load_data import easy_load_from_onnx
from SplitToChilds.runtime import FilterChildInput,FilterChildParams
from SplitToChilds.transfer import ModelNames
import importlib
import time  
import pynvml
from SplitToChilds.experience.export2lib import LoadLib

def GetGPUMemoryHandle():
    pynvml.nvmlInit() 
    return pynvml.nvmlDeviceGetHandleByIndex(1)


def GetGPUUsed(handle)->float:
    '''
    return MIB
    '''
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used/1048576  # MB

def RunWholeOnnxModel(model_name:str,input:dict, shape_dict:dict, driver=drivers.GPU(),onnx_download_url=OnnxModelUrl.Default,validate_download=False,count=5):
    gpuHandle=GetGPUMemoryHandle()

    # (N,3,224,224)——need to set input size for tvm
    mod, params, _ =easy_load_from_onnx(model_name,shape_dict,download_url=onnx_download_url,validate_download=validate_download)

    start_memory=GetGPUUsed(gpuHandle)
    with tvm.transform.PassContext(opt_level=0):
        lib = relay.build(mod, target=driver.target, params=params)
    module = graph_executor.GraphModule(lib["default"](driver.device))

    # tvm model
    for k, v in input.items(): 
        module.set_input(k, v)
    module.run()
    output = module.get_output(0).numpy().flatten()

    start=time.time()
    
    for _ in range(count):
        for k, v in input.items(): 
            module.set_input(k, v)
        module.run()
    output = module.get_output(0).numpy().flatten()
    avg_time=(time.time()-start)/count
    #print(output[:10])
    return output, params,avg_time,GetGPUUsed(gpuHandle)-start_memory

def RunWholeModelByFunction(model_name:str,input:dict,params:dict=None,driver=drivers.GPU(),allow_lib=True,count=5):
    pythonLib=importlib.import_module("ModelFuntionsPython.raw.{}".format(model_name))

    gpuHandle=GetGPUMemoryHandle()
    start_memory=GetGPUUsed(gpuHandle)
    lib=None
    if allow_lib:
        lib,_ = LoadLib(model_name,driver,-1)

    if lib is not None:
        print("  > skip build Lib")
    else:
        ir_module = tvm.IRModule.from_expr(getattr(pythonLib,ModelNames[model_name])())
        with tvm.transform.PassContext(opt_level=0):
            lib = relay.build(ir_module, driver.target, params=params)
    module = graph_executor.GraphModule(lib["default"](driver.device))

    for k, v in input.items():
        module.set_input(k, v)
    module.run()
    output = module.get_output(0).numpy().flatten()

    start=time.time()
    for _ in range(count):
        for k, v in input.items(): 
            module.set_input(k, v)
        module.run()
    output = module.get_output(0).numpy().flatten()
    avg_time=(time.time()-start)/count

    #print(output[:10])
    return output,avg_time,GetGPUUsed(gpuHandle)-start_memory

def RunAllChildModelSequentially(model_name:str,input:dict,params:dict=None,params_dict:dict=None, driver=drivers.GPU(),allow_lib=True,count=5):
    avg_time_list=[]
    memory_list=[]

    output=None
    for idx in range(len(params_dict)):
        print("--run model-%d:"%idx) 
        
        output,time_cost,memory_cost=RunChildModelByIdx(model_name,idx,input,params,params_dict,driver,output,allow_lib,count)
        avg_time_list.append(time_cost),memory_list.append(memory_cost)
    return output,avg_time_list,memory_list

def RunChildModelByIdx(model_name:str,idx:int,input:dict,params:dict=None,params_dict:dict=None, driver=drivers.GPU(),pre_output=None,allow_lib=True,count=5):
    pythonLib=importlib.import_module("ModelFuntionsPython.childs.{}".format(model_name))

    # new_input, new_params = FilterChildInput(params_dict,idx,input,pre_output),FilterChildParams(params_dict,idx,params)
    # ir_module = tvm.IRModule.from_expr(getattr(pythonLib,ModelNames[model_name]+"_"+str(idx))())
    # with tvm.transform.PassContext(opt_level=0):
    #     lib = relay.build(ir_module, driver.target, params=new_params)
    # module = graph_executor.GraphModule(lib["default"](driver.device))

    gpuHandle=GetGPUMemoryHandle()
    start_memory=GetGPUUsed(gpuHandle)

    lib=None
    if allow_lib:
        lib,_ = LoadLib(model_name,driver,idx)

    if lib is not None:
        print("  > skip build Lib")
    else:
        if params is None:
            print("error: need params.")
            return None
        new_params = FilterChildParams(params_dict,idx,params)
        ir_module = tvm.IRModule.from_expr(getattr(pythonLib,ModelNames[model_name]+"_"+str(idx))())
        with tvm.transform.PassContext(opt_level=0):
            lib = relay.build(ir_module, driver.target, params=new_params)
    module = graph_executor.GraphModule(lib["default"](driver.device))

    new_input = FilterChildInput(params_dict,idx,input,pre_output)
    for k, v in new_input.items():
        module.set_input(k, v)

    module.run()

    start=time.time()
    for _ in range(count):
        for k, v in new_input.items(): 
            module.set_input(k, v)
        module.run()

    return module.get_output(0).numpy(),(time.time()-start)/count,GetGPUUsed(gpuHandle)-start_memory