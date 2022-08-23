import netron
from config import Config

#onnx_path = "../onnxs/easy_model/easy_model.onnx"
onnx_path = Config.ModelSavePathName("FasterRCNN-10")
netron.start(onnx_path)
