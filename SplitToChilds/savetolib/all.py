from SplitToChilds.support import SupportedModels
from SplitToChilds.experience import export2lib
import torch
from config import Config
import drivers
from load_data import easy_load_from_onnx

validate_local_onnx_file=False
mydrivers=[drivers.GPU(),drivers.CPU()]

if __name__ == "__main__":
    print("pre-save TVM-Lib:")
    for model_name, config in SupportedModels.items():
        print("\n==>start to cache model:",model_name)

        input_shape = config["input_shape"]
        input_name = config["input_name"]

        input_data = torch.rand(*input_shape)
        input = {input_name: input_data}
        shape_dict = {input_name: input_data.shape}

        _, params, _ = easy_load_from_onnx(model_name,shape_dict,download_url=config["onnx_download_url"],validate_download=validate_local_onnx_file)
        for current_driver in mydrivers:
            export2lib.ExportModelToLib(model_name,params,Config.ModelParamsFile(model_name=model_name),driver=current_driver)