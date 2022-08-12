import netron
from config import Config

#onnx_path = "../onnxs/easy_model/easy_model.onnx"
onnx_path = Config.ModelSavePathName("easy_model")
netron.start(onnx_path)
