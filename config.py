import os

class Config:
    # static property
    ProjectRootPath=os.path.dirname(os.path.abspath(__file__))
    OnnxSavePath=os.path.join(ProjectRootPath,"onnxs")

    @staticmethod
    def ModelSavePathName(name)->str:
        '''
        name is given when you create the data
        '''
        return os.path.join(Config.OnnxSavePath,name,name+".onnx")

    @staticmethod
    def ModelSaveDataPathName(name)->str:
        '''
        name is given when you create the data
        '''
        return os.path.join(Config.OnnxSavePath,name,"data.json")