from GenerateModels.easy_model import get_ir_module
from gen_irmodule import MyParser

if __name__ == "__main__":
    mod = get_ir_module()
    parse = MyParser(mod)
    # parse.parse_params_with_module()

    # line = "def @main(%part1_input: Tensor[(4, 3, 14, 14), float32], %weight1: Tensor[(1, 3, 4, 4), float32], %bias1: Tensor[(1), float32], %add1: Tensor[(4, 3, 14, 14), float32], %weight2: Tensor[(1, 3, 4, 4), float32], %bias2: Tensor[(1), float32]) {"
    # bottoms, tops = parse.parse_params_with_text(line)
    # print(bottoms)
    # print(tops)

    txt_file_path = "/home/onceas/wanna/ModelSplit/AutoGenIRModule/text/resnet50.txt"
    py_file_path = txt_file_path.replace("txt", "py").replace("text", "pyfile")
    parse.parse_with_text(txt_file_path)
    module_name = "ResnetModule"
    parse.export_py_file(module_name, py_file_path)
