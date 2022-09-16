from SplitToChilds.moduleOperate import MyParser
from SplitToChilds.transfer import ModelNames
from config import Config

def SplitModel(model_name):
    rawModelFunctionTextPath = Config.RawModelFunctionsTextSavePathName(model_name)
    parse = MyParser(rawModelFunctionTextPath)
    
    parse.ParseWithFunctionText()
    parse.ExportToPythonFile(ModelNames[model_name], Config.RawModelFunctionsPythonSavePathName(model_name),clear=True)
    parse.BuildGraph()
    convergenceNodes = parse.FindConvergencePoint()
    for node in convergenceNodes:          
        node.PrintNode()
        
    #funtionTextPaths, paramsFileSavePath = parse.SplitToFunctionsTextFile([convergenceNodes[5], convergenceNodes[9], convergenceNodes[14]],aimDir=Config.ChildModelFunctionsTextSaveFold(model_name))
    funtionTextPaths, paramsFileSavePath = parse.SplitToFunctionsTextFile(convergenceNodes,aimDir=Config.ChildModelFunctionsTextSaveFold(model_name))
    for idx, funtionTextPath in enumerate(funtionTextPaths):
        parse = MyParser(funtionTextPath)
        parse.ParseWithFunctionText()
        parse.ExportToPythonFile(ModelNames[model_name]+"_"+str(idx), Config.ChildModelFunctionsPythonSavePathName(model_name),clear=True if idx==0 else False)
    return paramsFileSavePath


if __name__ == "__main__":
    model_name = "resnet50"
    SplitModel(model_name)
