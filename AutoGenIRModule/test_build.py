from GenerateModels.easy_model import get_ir_module
from gen_irmodule import MyParser

txt_to_class = {
    "googlenet": "GoogleNetModule",
    "resnet50": "ResnetModule",
    "easy_model": "EasyModule",
    "yolov2": "YoloModule",
    "squeezenet1": "SqueezeNetModule",
    "mobilenetv2": "MobileNetModule",
    "vgg19": "Vgg19Module",
}

if __name__ == "__main__":
    mod = get_ir_module()
    # parse.parse_params_with_module()

    # line = "def @main(%part1_input: Tensor[(4, 3, 14, 14), float32], %weight1: Tensor[(1, 3, 4, 4), float32], %bias1: Tensor[(1), float32], %add1: Tensor[(4, 3, 14, 14), float32], %weight2: Tensor[(1, 3, 4, 4), float32], %bias2: Tensor[(1), float32]) {"
    # bottoms, tops = parse.parse_params_with_text(line)
    # print(bottoms)
    # print(tops)

    # txt_name = "resnet50"
    txt_name = "easy_model"
    txt_file_path = "/home/onceas/wanna/ModelSplit/AutoGenIRModule/text/{}.txt".format(
        txt_name)
    parse = MyParser(mod, txt_file_path)
    parse.parse_with_text(txt_file_path)
    module_name = txt_to_class[txt_name]
    # parse.export_py_file(module_name, py_file_path)
    parse.build_graph()
    nodes = parse.find_convergence_point()
    # for _, node in enumerate(nodes):
    # node.print_self()
    nodes[-5].print_self()
    file_list, params_file_path = parse.split_txt_file([nodes[-5]])
    for idx, file_txt_path in enumerate(file_list):
        parse = MyParser(mod, file_txt_path)
        parse.parse_with_text(file_txt_path)
        py_file_path = file_txt_path.replace(
            "txt", "py").replace("text", "pyfile")
        print(py_file_path)
        parse.export_py_file(module_name, py_file_path)
