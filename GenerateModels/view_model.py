import netron
from config import Config

# onnx_path = "../onnxs/easy-model/easy-model.onnx"
netron.start(Config.ModelSavePathName("easy-model"))
