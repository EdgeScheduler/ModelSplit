from audioop import avg
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


def RunWholeModelByFunction(model_name:str,input:dict,params:dict,driver=drivers.GPU(),count=5):
    pythonLib=importlib.import_module("ModelFuntionsPython.raw.{}".format(model_name))

    gpuHandle=GetGPUMemoryHandle()
    # print("params:", params.keys())
    ir_module = tvm.IRModule.from_expr(getattr(pythonLib,ModelNames[model_name])())

    start_memory=GetGPUUsed(gpuHandle)
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

def RunAllChildModelSequentially(model_name:str,input:dict,params:dict,params_dict:dict=None, driver=drivers.GPU(),count=5):
    pythonLib=importlib.import_module("ModelFuntionsPython.childs.{}".format(model_name))
    
    gpuHandle=GetGPUMemoryHandle()
    avg_time_list=[]
    memory_list=[]
    output=None
    
    start=GetGPUUsed(gpuHandle)
    for idx in range(len(params_dict)):
        print("--run model-%d:"%idx) 
        new_input, new_params = FilterChildInput(params_dict,idx,input,output),FilterChildParams(params_dict,idx,params)
        # print("input:",list(new_input.keys()))
        # print("params:",list(new_params.keys()))
        
        ir_module = tvm.IRModule.from_expr(getattr(pythonLib,ModelNames[model_name]+"_"+str(idx))())
        # start_memory=GetGPUUsed(gpuHandle)
        time.sleep(3)
        with tvm.transform.PassContext(opt_level=0):
            lib = relay.build(ir_module, driver.target, params=new_params)
        module = graph_executor.GraphModule(lib["default"](driver.device))

        for k, v in new_input.items():
            module.set_input(k, v)
        module.run()
        output = module.get_output(0).numpy()

        start=time.time()
        for _ in range(count):
            for k, v in new_input.items(): 
                module.set_input(k, v)
            module.run()
        output = module.get_output(0).numpy()
        avg_time_list.append((time.time()-start)/count)
        # memory_list.append(GetGPUUsed(gpuHandle)-start_memory)
        memory_list.append(GetGPUUsed(gpuHandle))
    # print("output=>", output.flatten()[:10])
    return output,avg_time_list,memory_list

def RunChildModelByIdx(model_name:str,idx:int,input:dict,params:dict,params_dict:dict=None, driver=drivers.GPU(),pre_output=None,count=5):
    pythonLib=importlib.import_module("ModelFuntionsPython.childs.{}".format(model_name))

    new_input, new_params = FilterChildInput(params_dict,idx,input,pre_output),FilterChildParams(params_dict,idx,params)
    ir_module = tvm.IRModule.from_expr(getattr(pythonLib,ModelNames[model_name]+"_"+str(idx))())
    with tvm.transform.PassContext(opt_level=0):
        lib = relay.build(ir_module, driver.target, params=new_params)
    module = graph_executor.GraphModule(lib["default"](driver.device))

    for k, v in new_input.items():
        module.set_input(k, v)

    module.run()

    output = module.get_output(0).numpy()
    return output

def RunChildModelByRange(model_name:str,start:int,end:int,input:dict,params:dict,params_dict:dict=None, driver=drivers.GPU(),pre_output=None,count=5):
    pythonLib=importlib.import_module("ModelFuntionsPython.childs.{}".format(model_name))
    
    child_count=len(params_dict)
    if end is None or end>child_count:
        end=child_count
    elif end<0:
        end=child_count+end

    if start<0:
        start=child_count+start

    if not (end>start and start<child_count and start>=0):
        print("error: bad index, out of child-range")
        return

    # print run tips
    if start==end-1:
        print("--run model (%d):"%(start))
    else:
        print("--run model (%d=>%d):"%(start,end-1))

    output=pre_output
    for idx in range(start,end):
        
        # print("input:",list(new_input.keys()))
        # print("params:",list(new_params.keys()))
        new_input, new_params = FilterChildInput(params_dict,idx,input,output),FilterChildParams(params_dict,idx,params)
        ir_module = tvm.IRModule.from_expr(getattr(pythonLib,ModelNames[model_name]+"_"+str(idx))())
        with tvm.transform.PassContext(opt_level=0):
            lib = relay.build(ir_module, driver.target, params=new_params)
        module = graph_executor.GraphModule(lib["default"](driver.device))

        for k, v in new_input.items():
            module.set_input(k, v)
        module.run()
        output = module.get_output(0).numpy()
    # print("output=>", output.flatten()[:10])
    return output