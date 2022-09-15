import netron
from config import Config

#onnx_path = "../onnxs/easy_model/easy_model.onnx"
onnx_path = Config.ModelSavePathName("yolov5m6")
netron.start(onnx_path)
