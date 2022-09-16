from lib2to3.pgen2 import driver
from SplitToChilds.support import SupportedModels
from SplitToChilds.experience import runmodule
import torch
from config import Config
import drivers

validate_local_onnx_file=False
driver=drivers.GPU()

if __name__ == "__main__":
    print("run validate:")
    for model_name, config in SupportedModels.items():
        print("\n==>start to validate model:",model_name)

        input_shape = config["input_shape"]
        input_name = config["input_name"]

        input_data = torch.rand(*input_shape)
        input = {input_name: input_data}
        shape_dict = {input_name: input_data.shape}

        _, params = runmodule.RunWholeOnnxModel(model_name,input,shape_dict,driver=driver,onnx_download_url=config["onnx_download_url"],validate_download=validate_local_onnx_file)
        runmodule.RunWholeModelByFunction(model_name,input,params)
        runmodule.RunAllChildModelSequentially(model_name,input,params,Config.ModelParamsFile(model_name=model_name))