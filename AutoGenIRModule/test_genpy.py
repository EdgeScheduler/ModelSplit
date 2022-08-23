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
    parse = MyParser(mod)
    # parse.parse_params_with_module()

    # line = "def @main(%part1_input: Tensor[(4, 3, 14, 14), float32], %weight1: Tensor[(1, 3, 4, 4), float32], %bias1: Tensor[(1), float32], %add1: Tensor[(4, 3, 14, 14), float32], %weight2: Tensor[(1, 3, 4, 4), float32], %bias2: Tensor[(1), float32]) {"
    # bottoms, tops = parse.parse_params_with_text(line)
    # print(bottoms)
    # print(tops)

    # txt_name = "resnet50"
    txt_name = "vgg19"
    txt_file_path = "/home/onceas/wanna/ModelSplit/AutoGenIRModule/text/{}.txt".format(
        txt_name)
    py_file_path = txt_file_path.replace("txt", "py").replace("text", "pyfile")
    parse.parse_with_text(txt_file_path)
    module_name = txt_to_class[txt_name]
    parse.export_py_file(module_name, py_file_path)
