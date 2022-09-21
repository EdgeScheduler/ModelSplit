import tvm
import tvm.relay as relay
import drivers
from SplitToChilds.runtime import FilterChildParams,FilterChildInput
from SplitToChilds.transfer import ModelNames
from tvm.contrib import graph_executor
import importlib
import os
import json
from config import Config
from typing import Dict,Tuple

def ExportModelToLib(model_name:str,input:dict,params:dict,params_dict:dict=None, driver=drivers.GPU()):
    # save child-models
    pythonLib=importlib.import_module("ModelFuntionsPython.childs.{}".format(model_name))
    output=None
    for idx in range(len(params_dict)):
        new_params = FilterChildParams(params_dict,idx,params)
        
        ir_module = tvm.IRModule.from_expr(getattr(pythonLib,ModelNames[model_name]+"_"+str(idx))())
        with tvm.transform.PassContext(opt_level=0):
            lib = relay.build(ir_module, driver.target, params=new_params)

        module = graph_executor.GraphModule(lib["default"](driver.device))
        new_input = FilterChildInput(params_dict,idx,input,output)
        StoreLib(lib,model_name,new_input,driver,idx)
        for k, v in new_input.items():
            module.set_input(k, v)

        module.run()
        output = module.get_output(0).numpy()
        

    # save whole-model
    pythonLib=importlib.import_module("ModelFuntionsPython.raw.{}".format(model_name))
    ir_module = tvm.IRModule.from_expr(getattr(pythonLib,ModelNames[model_name])())
    with tvm.transform.PassContext(opt_level=0):
        lib = relay.build(ir_module, driver.target, params=params)
        StoreLib(lib,model_name,input,driver,-1)

def StoreLib(lib, model_name,input_dict:dict,driver: drivers.DeviceDriver,idx):
    lib_path, input_json_path=Config.TvmLibSavePathByName(model_name,driver.target,idx)
    if idx>=0:
        print("=> store %s-%d to %s, %s"%(model_name,idx,lib_path,input_json_path))
    else:
        print("=> store %s to %s, %s"%(model_name,lib_path,input_json_path))
    lib.export_library(lib_path)

    with open(input_json_path,'w') as fp:
        json.dump({k: str(tuple(v.shape)) for k,v in input_dict.items()},fp,indent=4)

def LoadLib(model_name,driver: drivers.DeviceDriver,idx):
    lib_path,input_json_path=Config.TvmLibSavePathByName(model_name,driver.target,idx)
    if os.path.exists(lib_path) and os.path.exists(input_json_path):
        return tvm.runtime.load_module(lib_path),LoadInputDict(model_name,driver,idx)
    else:
        return None,None

def LoadInputDict(model_name,driver: drivers.DeviceDriver,idx)->Dict[str, Tuple]:
    _,input_json_path=Config.TvmLibSavePathByName(model_name,driver.target,idx)
    if os.path.exists(input_json_path):
        input_dict={}
        with open(input_json_path,'r') as fp:
            input_dict = json.load(fp)
            for k in input_dict:
                input_dict[k]=eval(input_dict[k])
        return input_dict
    else:
        return None