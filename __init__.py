import os
from config import Config

os.makedirs(Config.OnnxSaveFold,exist_ok=True)
os.makedirs(Config.RawModelFunctionsTextSaveFold,exist_ok=True)
os.makedirs(Config.ChildsModelFunctionsTextSaveFold,exist_ok=True)
os.makedirs(Config.RawModelFunctionsPythonSaveFold,exist_ok=True)
os.makedirs(Config.ChildsModelFunctionsPythonSaveFold,exist_ok=True)
os.makedirs(Config.TVMLibSaveFold,exist_ok=True)
os.makedirs(os.path.join(Config.ProjectRootFold, "Benchmark/timecost",exist_ok=True))