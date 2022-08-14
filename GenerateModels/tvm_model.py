import onnx
import tvm.relay as relay
from tvm.contrib import graph_executor
import tvm


def get_tvm_model(onnx_path, shape_dict, target, dev):
    print(onnx_path)
    onnx_model = onnx.load(onnx_path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    # todo opt_level
    with tvm.transform.PassContext(opt_level=0):
        lib = relay.build(mod, target=target, params=params)

    module = graph_executor.GraphModule(lib["default"](dev))
    return module, params
