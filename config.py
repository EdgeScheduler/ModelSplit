import os


class Config:
    # static property
    ProjectRootPath = os.path.dirname(os.path.abspath(__file__))
    OnnxSavePath = os.path.join(ProjectRootPath, "onnxs")
    TestDataCount = 10

    @staticmethod
    def ModelSavePathName(name) -> str:
        '''
        name is given when you create the data. return "$project_path/onnxs/$name/$name.onnx"
        '''
        return os.path.join(Config.OnnxSavePath, name, name+".onnx")

    @staticmethod
    def TvmLibSavePathName(name, target, batch) -> str:
        '''
        name is given when you create the data. return "$project_path/onnxs/$name/$name.tar"
        '''
        return os.path.join(Config.OnnxSavePath, name, "{}-{}-{}.tar".format(name, target, batch))

    @staticmethod
    def ModelSaveDataPathName(name) -> str:
        '''
        name is given when you create the data. return "$project_path/onnxs/$name/data.json"
        '''
        return os.path.join(Config.OnnxSavePath, name, "data.json")


class OnnxModelUrl:
    '''
    record onnx models URL which can be used to download from internet.
    '''
    Default = ""
    Resnet50 = ""
