# test easy-model
import tvm
from tvm import relay
import onnx
import drivers
from config import Config
import load_data

name="easy-model-fish"
shape_dict={"input_edge": (4,3,14,14)}

irModule,params= load_data.easy_load_from_onnx(name,shape_dict)

# easy_model=onnx.load(Config.ModelSavePathName(name))

# mod, params = relay.frontend.from_onnx(easy_model, shape_dict)

# irModule = relay.transform.InferType()(mod)                         # 补充onnx导入后缺失的信息

# print(irModule,params)

# mod["main"]: tvm.relay.function.Function

# lib = relay.build(irModule, target=drivers.CPU.target, params=params)
print(type(irModule["main"]))
# shape_dict = {v.name_hint: v.checked_type for v in mod["main"].params}
# print(shape_dict)
# print(irModule)