import tvm
import tvm.relay as relay
import drivers
from SplitToChilds.runtime import FilterChildParams
from SplitToChilds.transfer import ModelNames
import importlib
import os
from config import Config

def ExportModelToLib(model_name:str,params:dict,params_dict:dict=None, driver=drivers.GPU()):
    # save child-models
    pythonLib=importlib.import_module("ModelFuntionsPython.childs.{}".format(model_name))
    for idx in range(len(params_dict)):
        new_params = FilterChildParams(params_dict,idx,params)
        
        ir_module = tvm.IRModule.from_expr(getattr(pythonLib,ModelNames[model_name]+"_"+str(idx))())
        with tvm.transform.PassContext(opt_level=0):
            lib = relay.build(ir_module, driver.target, params=new_params)
            StoreLib(lib,model_name,driver,idx)

    # save whole-model
    pythonLib=importlib.import_module("ModelFuntionsPython.raw.{}".format(model_name))
    ir_module = tvm.IRModule.from_expr(getattr(pythonLib,ModelNames[model_name])())
    with tvm.transform.PassContext(opt_level=0):
        lib = relay.build(ir_module, driver.target, params=params)
        StoreLib(lib,model_name,driver,-1)

def StoreLib(lib, model_name,driver: drivers.DeviceDriver,idx):
    lib_path=Config.TvmLibSavePathByName(model_name,driver.target,idx)
    if idx>=0:
        print("=> store %s-%d to %s"%(model_name,idx,lib_path))
    else:
        print("=> store %s to %s"%(model_name,lib_path))
    lib.export_library(lib_path)

def LoadLib(model_name,driver: drivers.DeviceDriver,idx):
    lib_path=Config.TvmLibSavePathByName(model_name,driver.target,idx)
    if os.path.exists(lib_path):
        return tvm.runtime.load_module(lib_path)
    else:
        return None