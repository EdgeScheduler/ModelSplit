import os
from config import Config
import sys

print(sys.argv[0])
sys.path.append(sys.argv[0])

os.makedirs(Config.OnnxSaveFold, exist_ok=True)
os.makedirs(Config.RawModelFunctionsTextSaveFold, exist_ok=True)
os.makedirs(Config.ChildsModelFunctionsTextSaveFold, exist_ok=True)
os.makedirs(Config.RawModelFunctionsPythonSaveFold, exist_ok=True)
os.makedirs(Config.ChildsModelFunctionsPythonSaveFold, exist_ok=True)
