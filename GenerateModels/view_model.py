import netron
from config import Config

<<<<<<< HEAD
#onnx_path = "../onnxs/easy_model/easy_model.onnx"
onnx_path = Config.ModelSavePathName("easy_model")
netron.start(onnx_path)
=======
# onnx_path = "../onnxs/easy-model/easy-model.onnx"
netron.start(Config.ModelSavePathName("easy-model"))
>>>>>>> b493645677ff6e5755ae7159ba2828e30e47862d
