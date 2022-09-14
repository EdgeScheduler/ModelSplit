from AutoGenIRModule.gen_irmodule import Layer, MyParser
from config import Config
from AutoGenIRModule.test_split.common import ModelNames

# txt_to_class = {
#     "googlenet": "GoogleNetModule",
#     "resnet50": "ResnetModule",
#     "easy_model": "EasyModule",
#     "yolov2": "YoloModule",
#     "squeezenet1": "SqueezeNetModule",
#     "mobilenetv2": "MobileNetModule",
#     "mobilenetv2_back": "MobileNetModule",
#     "vgg19": "Vgg19Module",
# }

model_name = "easy_model"

def splitModel():
    rawModelFunctionTextPath = Config.RawModelFunctionsTextSavePathName(model_name)
    parse = MyParser(rawModelFunctionTextPath)
    
    parse.ParseWithFunctionText(rawModelFunctionTextPath)
    parse.ExportToPythonFile(model_name, Config.RawModelFunctionsPythonSavePathName(model_name),clear=True)
    parse.BuildGraph()
    convergenceNodes = parse.FindConvergencePoint()
    for node in convergenceNodes:          
        node.PrintNode()
        
    funtionTextPaths, paramsFileSavePath = parse.SplitToFunctionsTextFile([convergenceNodes[-4]],aimDir=Config.ChildModelFunctionsTextSaveFold(model_name))
    for idx, funtionTextPath in enumerate(funtionTextPaths):
        parse = MyParser(funtionTextPath)
        parse.ParseWithFunctionText(funtionTextPath)
        parse.ExportToPythonFile(ModelNames[model_name]+"_"+str(idx), Config.ChildModelFunctionsPythonSavePathName(model_name),clear=True if idx==0 else False)
    return paramsFileSavePath


if __name__ == "__main__":
    splitModel()
