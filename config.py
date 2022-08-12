import os

class Config:
    # static property
    ProjectRootPath=os.path.dirname(os.path.abspath(__file__))
    OnnxSavePath=os.path.join(ProjectRootPath,"onnxs")

    @staticmethod
    def ModelSavePathName(name)->str:
        '''
        name is given when you create the data. return "$project_path/onnxs/$name/$name.onnx"
        '''
        return os.path.join(Config.OnnxSavePath,name,name+".onnx")

    @staticmethod
    def ModelSaveDataPathName(name)->str:
        '''
        name is given when you create the data. return "$project_path/onnxs/$name/data.json"
        '''
        return os.path.join(Config.OnnxSavePath,name,"data.json")