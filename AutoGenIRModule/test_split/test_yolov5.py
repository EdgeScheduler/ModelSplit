from GenerateModels.easy_model import get_ir_module
from AutoGenIRModule.gen_irmodule import MyParser,Layer


txt_to_class = {
    "googlenet": "GoogleNetModule",
    "resnet50": "ResnetModule",
    "easy_model": "EasyModule",
    "yolov2": "YoloModule",
    "yolov5m6": "Yolov5Module",
    "squeezenet1": "SqueezeNetModule",
    "mobilenetv2": "MobileNetModule",
    "mobilenetv2_back": "MobileNetModule",
    "vgg19": "Vgg19Module",
}


def gen_split_model():
    mod = get_ir_module()

    txt_name = "yolov5m6"
    txt_file_path = "/home/onceas/wanna/ModelSplit/AutoGenIRModule/text/{}.txt".format(
        txt_name)
    parse = MyParser(mod, txt_file_path)

    py_file_path = txt_file_path.replace("txt", "py").replace("text", "pyfile")
    parse.ParseWithFunctionText(txt_file_path)
    # module_name = txt_to_class[txt_name]
    # parse.export_py_file(module_name, py_file_path)
    parse.BuildGraph()
    # parse.bfs()
    convergence_nodes = parse.FindConvergencePoint()
    for _, node in enumerate(convergence_nodes):
        node.Print()
    print("len=", len(convergence_nodes))
    file_path_list, params_file_path = parse.SplitToFunctionsTextFile(
        [convergence_nodes[10], convergence_nodes[-10], convergence_nodes[-4]])
    for idx, file_path in enumerate(file_path_list):
        parse = MyParser(file_path)
        py_file_path = file_path.replace(
            "txt", "py").replace("text", "pyfile")
        parse.ParseWithFunctionText(file_path)
        module_name = txt_to_class[txt_name]+"_"+str(idx)
        parse.ExportToPythonFile(module_name, py_file_path)
    return params_file_path


if __name__ == "__main__":
    gen_split_model()
