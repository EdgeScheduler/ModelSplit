import os
from typing import Dict,List
import json

class Config:
    # static path property
    ProjectRootFold = os.path.dirname(os.path.abspath(__file__))

    # onnx-model save-path
    OnnxSaveFold = os.path.join(ProjectRootFold, "Onnxs")

    # model-functions text save-path
    RawModelFunctionsTextSaveFold = os.path.join(ProjectRootFold, "ModelFuntionsText/raw")
    ChildsModelFunctionsTextSaveFold= os.path.join(ProjectRootFold, "ModelFuntionsText/childs")

    # model-functions Python-file save-path
    RawModelFunctionsPythonSaveFold = os.path.join(ProjectRootFold, "ModelFuntionsPython/raw")
    ChildsModelFunctionsPythonSaveFold= os.path.join(ProjectRootFold, "ModelFuntionsPython/childs")

    TestDataCount = 10 

    @staticmethod
    def RawModelFunctionsTextSavePathName(model_name)->str:
        '''
        name is given when you use raw model functions text from disk, you may create this file by print-copy. Return "$project_path/ModelFuntionsText/raw/$model_name.txt"
        '''
        return os.path.join(Config.RawModelFunctionsTextSaveFold,model_name+".txt")

    @staticmethod
    def RawModelFunctionsPythonSavePathName(model_name)->str:
        '''
        name is given when you use raw model functions Python-file from disk, you may create this file by print-copy. Return "$project_path/ModelFuntionsPython/raw/$model_name.py"
        '''
        return os.path.join(Config.RawModelFunctionsPythonSaveFold,model_name+".py")

    @staticmethod
    def ModelParamsFile(model_name)->Dict[int,List[dict]]:
        '''
        return convert "$project_path/ModelFuntionsText/childs/$model_name/params.json" to dict
        '''
        jsonFilePath=os.path.join(Config.ChildModelFunctionsTextSaveFold(model_name),"params.json")
        if not os.path.exists(jsonFilePath):
            return None
        
        with open(jsonFilePath,"r") as fp:
            try:
                return json.load(fp)
            except Exception as ex:
                print("error:",ex)
                return None
    
    @staticmethod
    def ChildModelFunctionsPythonSavePathName(model_name)->str:
        '''
        name is given when you use child-model functions Python-file from disk, you may create this file by print-copy. Return "$project_path/ModelFuntionsPython/childs/$model_name.py"
        '''
        return os.path.join(Config.ChildsModelFunctionsPythonSaveFold,model_name+".py")

    @staticmethod
    def ChildModelFunctionsTextSaveFold(model_name)->str:
        '''
        name is given when you use child-model functions Text-file fold from disk, you may create this file by print-copy. Return "$project_path/ModelFuntionsText/childs/$model_name/"
        '''
        return os.path.join(Config.ChildsModelFunctionsTextSaveFold,model_name)


    @staticmethod
    def ModelSavePathName(name) -> str:
        '''
        name is given when you create the data. Return "$project_path/Onnxs/$name/$name.onnx"
        '''
        return os.path.join(Config.OnnxSaveFold, name, name+".onnx")

    @staticmethod
    def TvmLibSavePathName(name, target, batch) -> str:
        '''
        name is given when you create the data. Return "$project_path/Onnxs/$name/$name.tar"
        '''
        return os.path.join(Config.OnnxSaveFold, name, "{}-{}-{}.tar".format(name, target, batch))

    @staticmethod
    def ModelSaveDataPathName(name) -> str:
        '''
        name is given when you create the data. Return "$project_path/Onnxs/$name/data.json"
        '''
        return os.path.join(Config.OnnxSaveFold, name, "data.json")