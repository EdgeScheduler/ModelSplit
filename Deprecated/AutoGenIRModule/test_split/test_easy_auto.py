import AutoGenIRModule
from GenerateModels.easy_model import get_ir_module
from AutoGenIRModule.gen_irmodule import MyParser
import numpy as np
from ModelUtils.model_utils import load_onnx_model, onnx2IRModule, build_lib, store_lib, get_lib
from config import Config
import time
from relayIR.relay_graph import construct_op_graph
import drivers
from AutoGenIRModule.pyfile.easy_model import EasyModule
from AutoGenIRModule.pyfile.easy_model_split.easy_model_0 import EasyModule_0
from AutoGenIRModule.pyfile.easy_model_split.easy_model_1 import EasyModule_1
from ModelAnalyze.easy_model_split_manual import whole_model, get_whole_model_params
from ModelUtils.params_utils import parse_params_file, filter_params
from test_easy import splitModel
from test_easy_compare import run_whole_model, run_split_model

if __name__ == "__main__":
    params_file_path = splitModel()
    run_whole_model()
    run_split_model(params_file_path)
