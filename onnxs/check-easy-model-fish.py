# test easy-model

import tvm
from tvm import relay
import onnx
import os
from config import Config

name="easy-model-fish"


easy_model=onnx.load(Config.ModelSavePathName(name))
mod, params = relay.frontend.from_onnx(easy_model)
mod=relay.transform.InferType()(mod)    # tvm.ir.module.IRModule

print(type(mod))

# shape_dict = {v.name_hint : v.checked_type for v in mod["main"].params}
# for label,type_name in shape_dict.items():
#     print(label,":",type_name)

# print(mod,params)