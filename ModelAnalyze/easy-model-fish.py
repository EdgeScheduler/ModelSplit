# test easy-model
import tvm
from tvm import relay
import onnx
import drivers
from config import Config

name="easy-model-fish"

easy_model=onnx.load(Config.ModelSavePathName(name))
shape_dict={"input_edge": (4,3,14,14)}
mod, params = relay.frontend.from_onnx(easy_model, shape_dict)

irModule = relay.transform.InferType()(mod)

print(irModule,params)

lib = relay.build(irModule, target=drivers.CPU.target, params=params)