# test easy-model

import tvm
from tvm import relay
import onnx
import os
from config import Config

name="easy-model-fish"

easy_model=onnx.load(Config.ModelSavePathName(name))
shape_dict={"input_edge": (4,3,14,14)}
mod, params = relay.frontend.from_onnx(easy_model, shape_dict)
model = relay.transform.InferType()(mod)


print(model,params)


# lib = relay.build(module, target=target_, params=params)