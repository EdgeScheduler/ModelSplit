import netron
from config import Config

#onnx_path = "../onnxs/easy_model/easy_model.onnx"
onnx_path = Config.ModelSavePathName("googlenet-7")
netron.start(
    "/home/onceas/wanna/ModelSplit/Deprecated/Onnxs/googlenet-7/googlenet-7.onnx")
