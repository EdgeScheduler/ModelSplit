# test easy-model

import tvm
from tvm import relay
import onnx
import os
from config import Config

name="easy-model"

easy_model=onnx.load(Config.ModelSavePathName(name))
shape_dict={}
mod, params = relay.frontend.from_onnx(easy_model, shape_dict)

print(mod)
print(params)