import os
import json
import onnx
import onnxruntime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelUtils.model_utils import load_onnx_model, onnx2IRModule, build_lib, store_lib, get_lib
import load_data
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor
import onnxruntime as ort
from config import Config
import time
import drivers
from relayIR.relay_graph import construct_op_graph
from queue import Queue
import re
import argparse


class Layer():
    def __init__(self, name, type, bottoms, tops, params, line):
        # %1
        self.name = name
        # nn.dense
        self.type = type
        # stride=[1,1]
        self.params = params
        #  = {call_3,call_4}
        self.bottoms = bottoms
        # {call_2} =
        self.tops = tops
        self.line = line

    def print_self(self):
        print("Layer-------------")
        print("name:{} type:{} params:{}".format(
            self.name, self.type, self.params))
        print("tops:{} bottoms:{}".format(self.tops, self.bottoms))
        print("Layer-------------")


class GraphNode:
    def __init__(self, layer):
        # self info
        self.layer = layer
        # pre & next
        self.pre = list()
        self.next = list()
        # in & out degree
        self.in_degree = 0
        self.out_degree = 0

    def add_next_node(self, node):
        self.out_degree += 1
        self.next.append(node)

    def add_pre_node(self, node):
        self.in_degree += 1
        self.pre.append(node)

    def print_self(self):
        print("GraphNode------------")
        print("node:", self.layer.print_self())
        print("pre:", [item.layer.name for _, item in enumerate(self.pre)])
        print("next:", [item.layer.name for _, item in enumerate(self.next)])
        print("in_degree:", self.in_degree)
        print("out_degree:", self.out_degree)
        print("GraphNode------------")


class MyParser:
    def __init__(self, ir_module, txt_file_path):
        self.ir_module = ir_module
        self.txt_file_path = txt_file_path
        # nn.func
        self.layer_list = list()
        self.replace_map = dict()
        # input
        self.net_inputs = None,
        self.net_input_shapes = None,
        # constant
        self.net_metas = list()
        self.net_meta_shapes = dict()
        self.net_meta_dtypes = dict()
        # output
        self.output_count = 0
        # graph header
        self.header = None
        self.nodes = dict()

    def build_graph(self):
        for idx, layer in enumerate(self.layer_list):
            if layer is None:
                continue
            self.nodes[idx] = GraphNode(layer)
        for idx, layer in enumerate(self.layer_list):
            if layer is None:
                continue
            node = self.nodes[idx]
            for _, bottom in enumerate(layer.bottoms):
                for b in bottom if isinstance(bottom, list) else [bottom]:
                    if "call_" in b:
                        call_num = int(float(b.strip("call_")))
                        pre_node = self.nodes[call_num]
                        node.add_pre_node(pre_node)
                        pre_node.add_next_node(node)
        self.header = self.nodes[0]

    def bfs(self):
        print("bfs:-----------------------------")
        q = Queue()
        q.put(self.header)
        while not q.empty():
            front = q.get()
            front.print_self()
            for _, next in enumerate(front.next):
                q.put(next)
        print("bfs:-----------------------------")

    def cul_pre_degree(self, node):
        q = Queue()
        q.put(node)
        in_total = 0
        out_total = 0
        flag = dict()
        while not q.empty():
            front = q.get()
            if front.layer.name in flag.keys():
                continue
            flag[front.layer.name] = True
            in_total += front.in_degree
            out_total += front.out_degree
            for _, pre in enumerate(front.pre):
                q.put(pre)
        return in_total, out_total-node.out_degree

    def check_convergence_point(self, node):
        if node.in_degree == 0 or node.out_degree == 0:
            return False
        if node.out_degree != 1 or node.next[0].in_degree != 1:
            return False
        in_total, out_total = self.cul_pre_degree(node)
        # print(in_total, out_total)
        return in_total == out_total

    def find_convergence_point(self):
        res = list()
        for k, v in self.nodes.items():
            # print("k==", v.layer.name)
            if self.check_convergence_point(v):
                res.append(v)
        return res

    def parse_param(self, param):
        items = param.split(": ")
        name = items[0]
        tmp = items[1].split("Tensor[")[1].split("]")[0].split("), ")
        shape = tmp[0]+")"
        p_type = tmp[1]
        return {"name": name, "shape": shape, "type": p_type}

    def store_params(self, params_dict, params_file_path):
        # print(params_dict)
        new_dict = {}
        for k, v in params_dict.items():
            params = []
            for _, param in enumerate(v):
                params.append(self.parse_param(param))
            new_dict[k] = params
        with open(params_file_path, "w") as f:
            f.write(json.dumps(new_dict))

    def split_txt_file(self, nodes):
        file_name = self.txt_file_path.split("/")[-1].strip(".txt")
        dir_path = os.path.abspath(os.path.dirname(self.txt_file_path))

        split_file_dir = os.path.join(dir_path, file_name+"_split")
        if not os.path.exists(split_file_dir):
            os.mkdir(split_file_dir)
        idx = 0
        params_line = ""
        lines = list()
        with open(self.txt_file_path, "r") as rfp:
            for line in rfp:
                if "def" in line:
                    params_line = line
                lines.append(line)

        # split whole txt file
        flag = 0
        node = nodes[idx]
        file_lines = list()
        split_txt_files = dict()
        extra_input = None
        for line in lines[1:]:
            if flag == 0:
                if extra_input == None:
                    file_lines.append(params_line)
                else:
                    file_lines.append(params_line.replace(
                        "def @main(", "def @main("+"%call_"+extra_input.strip("%")+", "))
                    # print(params_line.replace(
                    # "def @main(", "def @main("+"%call_"+extra_input.strip("%")+", "))
                    extra_input = None
                flag = 1
            file_lines.append(line)
            # end here
            if node != None and node.layer.name+" = " in line:
                extra_input = node.layer.name+": " + \
                    line.split(") /* ty=")[-1].strip(" */;\n")

                tmp_line = line.replace(node.layer.name+" = ", "")
                file_lines[-1] = tmp_line
                file_lines.append("}")

                split_txt_files[idx] = file_lines
                flag = 0
                idx += 1
                file_lines = list()
                if (idx >= len(nodes)):
                    node = None
                else:
                    node = nodes[idx]
            elif line == "}" and node == None:
                split_txt_files[idx] = file_lines

        # fix input & params
        params_dict = dict()
        for k, v in split_txt_files.items():
            items = v[0].split("def @main(%")[1].split(" {\n")[0].split(", %")
            # print("items=", items)
            res = set()
            # if extra
            for _, item in enumerate(items):
                if "call_" in item.split(":")[0]:
                    res.add(item)
            # if exist
            for _, line in enumerate(v[1:]):
                for _, item in enumerate(items):
                    if item.split(":")[0] in line:
                        res.add(item)
            # print("res=", res)
            params_dict[k] = list(res)
            v[0] = "def @main(%"+", %".join(list(res))+" {\n"
        params_file_path = os.path.join(split_file_dir, "params.json")
        self.store_params(params_dict, params_file_path)

        # write split txt file
        file_list = list()
        for k, v in split_txt_files.items():
            split_file_path = os.path.join(
                split_file_dir, "{}_{}.txt".format(file_name, k))
            file_list.append(split_file_path)
            with open(split_file_path, "w") as wfp:
                for _, line in enumerate(v):
                    wfp.write(line)
        print("split txt file path:", file_list)
        print("params json file path:", params_file_path)
        return file_list, params_file_path

    def get_extra_input(self, node):
        # call_9
        # input_name = "call_"+node.layer.name.strip("%")
        # %9
        input_name = node.layer.name
        return input_name

    def split_model(self, nodes):
        pass

    def parse_params_with_module(self):
        mod_params = self.ir_module.functions.items()[0][1].params
        for param in mod_params:
            name = str(param.name_hint)
            tmp = str(param.type_annotation).replace(
                "Tensor[", "").replace("]", "").split("),")
            shape = tmp[0]+")"
            dtype = tmp[1]
            # print(name, shape, dtype)

    def judge_line_type(self, line):
        # self.net_inputs = list()
        # line = "  %114 = subtract(1f /* ty=float32 */, %90) /* ty=Tensor[(1, 12, 50), float32] */;"
        # step 0. get type
        # -1: unkonw
        # 0: "def @main(%INPUT__0..."
        # 1: "%0 = (%INPUT__0, %INPUT__1)"
        # 2: "%7 = %3.0" or "%99 = (%97, %98);"
        # 3: "%9 = nn.scale("
        # 4: "nn.fully_connected(%144..."
        # 5: "}"
        type = -1
        if "def" in line:
            type = 0
        elif "}" in line:
            type = 5
        elif " = " not in line:
            type = 4
        else:
            index = line.find('(')
            if index == -1:
                type = 2
            else:
                index2 = line.find('=')
                if index-index2 == 2:
                    type = 1
                else:
                    type = 3
        return type

    def parse_params_with_text(self, line):
        line = line.split("->")[0]+"{\n"
        # fix yolov5 onnx::Resize_730
        line = line.replace("::", "_")
        bottoms = [i.split(':')[0] for i in line.split("%")[1:]]
        shapes = [list(i.split(')')[0].split(', '))
                  for i in line.split("Tensor[(")[1:]]

        '''
        ['part1_input', 'weight1', 'bias1', 'add1', 'weight2', 'bias2']

        [['4', '3', '14', '14'], ['1', '3', '4', '4'], ['1'], [
        '4', '3', '14', '14'], ['1', '3', '4', '4'], ['1']]
        '''
        return bottoms, shapes

    def key_func(self, bottom, line):
        raw_index = line.find(bottom)
        if raw_index != -1:
            return raw_index
        if re.match(r"\d+f", bottom):
            raw_index = line.find(bottom)
        elif "Constant" in bottom:
            constant_bottom = "meta[relay.Constant][{}]".format(
                bottom.split('_')[1])
            raw_index = line.find(constant_bottom)
        elif "call_" not in bottom:
            # add(%225, %226)
            if line.find('('+bottom) != -1:
                raw_index = line.find('('+bottom)
            elif line.find(', '+bottom) != -1:
                raw_index = line.find(', '+bottom)
        else:
            raw_index = line.find("%"+bottom.strip("call_").split(".")[0])
        # print(bottom, raw_index)
        return raw_index

    def handle_constant_line(self, line):
        # meta[relay.Constant][0]
        # Constant_0
        constant_bottoms = [
            "Constant_"+i.split(']')[0] for i in line.split("meta[relay.Constant][")[1:]]
        # fix
        constant_shape = [i.split("ty=Tensor[")[1].split(
            "), ")[0]+')' for i in line.split("meta[relay.Constant]")[1:] if "ty=Tensor[" in i]
        constant_dtype = [i.split("ty=Tensor[")[1].split("), ")[1].split(
            ']')[0] for i in line.split("meta[relay.Constant]")[1:] if "ty=Tensor[" in i]
        self.net_metas += constant_bottoms
        # TODO: irmodule text need to print in detail
        if len(constant_shape) > 0:
            for i, const in enumerate(constant_bottoms):
                self.net_meta_shapes[const] = constant_shape[i]
                self.net_meta_dtypes[const] = constant_dtype[i]
        return constant_bottoms

    def parse_params(self, line):
        params = dict()
        s = line.split('=')
        # ['  %217 ', ' nn.conv2d(%216, %resnetv24_stage4_conv9_weight, padding', '[0, 0, 0, 0], channels', '2048, kernel_size', '[1, 1]);\n']
        for i in range(1, len(s)):
            # first =
            if s[i-1][-1] == ' ' and s[i][0] == ' ':  # %13 = nn.fully_connected
                continue
            # /* ty=Tensor[(1, 1, 50), float32] */;
            if "/*" in s[i-1] and "*/" in s[i]:
                continue

            param = s[i-1].split(", ")[-1]
            # padding = [1, 1, 1, 1]
            if s[i].split(", ")[0][0] == '[':
                value = s[i].split("]")[0]+']'
            else:  # channels = 160
                value = s[i].split(", ")[0].split(")")[0]
            # epsilon: 1e-08f     bias: -10000f
            if value[-1] == 'f':
                value = value.replace('f', '')
            params[param] = value
        return params

    def parse_with_text(self, relay_text_path):
        with open(relay_text_path, 'r') as f:
            for line in f:
                line = line.replace("::", "_")
                type = self.judge_line_type(line)
                # fix googlenet:conv1/7x7_s2_b_0 can't match input
                # line = line.replace("/", "_")
                # print(line, "-------", type)
                # step 1. process
                if type == -1:
                    raise

                elif type == 0:  # 0: "def @main(%INPUT__0..."
                    self.net_inputs, self.net_input_shapes = self.parse_params_with_text(
                        line)

                # 1: "  %0 = (%INPUT_0, %INPUT_1)"
                # 2: "  %7 = %3.0;\n" or "%99 = (%97, %98);"
                elif type == 1 or type == 2:
                    bottoms = [i.split(',')[0].split(';')[0].split(')')[0]
                               for i in line.split("%")[2:]]
                    bottoms = [
                        "call_"+bottom if bottom not in self.net_inputs else bottom for bottom in bottoms]
                    # %1
                    top = "call_"+line.split("%")[1].split(' =')[0]

                    for bottom in bottoms:
                        # %220 = %219.0
                        if '.' in bottom:
                            # print("bottom=", bottom)
                            # fix split index
                            idx = -1
                            for i, layer in enumerate(self.layer_list):
                                if layer == None:
                                    continue
                                if layer.tops[0] == bottom.split('.')[0]:
                                    idx = i
                                    break
                            if idx == -1:
                                # in input
                                continue
                            # 219
                            a = int(bottom.split('.')[0].strip("call_"))
                            # 0
                            i = int(bottom.split('.')[1])
                            if i == 0:
                                self.layer_list[idx].tops[0] += '.0'
                            else:
                                # TODO: ?
                                while i > len(self.layer_list[idx].tops)-1:
                                    self.layer_list[idx].tops.append(
                                        "call_"+str(idx)+'.'+str(len(self.layer_list[idx].tops)))
                        else:
                            continue

                    # handle meta[constant[0]]
                    if "meta[relay.Constant]" in line:
                        bottoms += self.handle_constant_line(line)

                    # reorder bottoms(as constant vs var is unordered)
                    bottoms.sort(
                        key=lambda bottom: self.key_func(bottom, line))

                    # top-bottoms replace map for type=3/4
                    self.replace_map[top] = bottoms
                    self.layer_list.append(None)

                elif type == 3 or type == 4:
                    # process link
                    # 3: "  %12 = nn.fused_qkv_attention(%10, %11, ..."
                    if type == 3:
                        '''
                          %0 = nn.conv2d(%part1_input, %weight1, strides=[2, 2], padding=[0, 0, 0, 0], channels=1, kernel_size=[4, 4]);
                        %0 
                        nn.conv2d
                        ['part1_input', 'weight1']
                        ['call_0']
                        %1 = nn.bias_add(%0, %bias1);
                        '''
                        name = line.split(" = ")[0].strip(" ")
                        type = line.split(" = ")[1].split("(")[0]
                        bottoms = [i.split(',')[0].split(')')[0]
                                   for i in line.split("%")[2:]]
                        # 1. part1_input
                        # 2. 0/1 → call_0/call_1
                        bottoms = [
                            "call_"+bottom if bottom not in self.net_inputs else bottom for bottom in bottoms]
                        tops = ["call_"+line.split("%")[1].split(' =')[0]]
                    elif type == 4:  # 4: "  nn.fully_connected(%144..."
                        ''''
                          add(%7, %8)
                        add
                        ['call_7', 'call_8']
                        ['call_output0']
                        '''
                        name = ""
                        type = line.split("(%")[0].strip()
                        bottoms = [i.split(',')[0].split(')')[0]
                                   for i in line.split("%")[1:]]
                        bottoms = [
                            "call_"+bottom if bottom not in self.net_inputs else bottom for bottom in bottoms]
                        tops = ["call_"+"output"+str(self.output_count)]
                        self.output_count += 1

                    # %134 = take(%72, 0 /* ty=int64 */, axis=1)
                    # %114 = subtract(1f /* ty=float32 */, %90)
                    # TODO: ?
                    # print("before:", bottoms)
                    tmp = [i.split(' ')[0].strip('f') for i in ''.join(line.split("(")[1:]).split(
                        ", ") if ' ' in i and i.split(' ')[0].strip('f').isdigit() == True]
                    # if len(tmp) > 0:
                    # print("tmp=", tmp)
                    # raise
                    # bottoms += tmp

                    # fix : %226 = multiply(1f, %resnetv24_dense0_bias);
                    consts = re.findall(
                        r"\(\d+f", line) + re.findall(r", \d+f", line)
                    for _, con in enumerate(consts):
                        bottoms += [con.strip("(").strip(", ")]
                    # print("after:", bottoms)

                    # handle meta[constant[0]]
                    if "meta[relay.Constant]" in line:
                        bottoms += self.handle_constant_line(line)

                    # reorder bottoms(as constant vs var is unordered)
                    bottoms.sort(
                        key=lambda bottom: self.key_func(bottom, line))

                    # 'call_1': ['call_0.0']
                    new_bottoms = list()
                    for bottom in bottoms:
                        if self.replace_map.get(bottom) != None:
                            new_bottoms.append(self.replace_map[bottom])
                        else:
                            new_bottoms.append(bottom)

                    # process param
                    params = self.parse_params(line)

                    l = Layer(name=name, type=type, bottoms=new_bottoms,
                              tops=tops, params=params, line=line)
                    self.layer_list.append(l)

                elif type == 5:
                    pass

    def export_py_file(self, module_name, relay_python_path):
        """
        export python file after parse IRModule text

        Parameters
        ----------
        :param relay_python_path: the file path of output python file
        """
        # print("------------------")
        # for index, layer in enumerate(self.layer_list):
        #     print("index=", index)
        #     '''
        #     index= 204
        #     name:%204 type:nn.conv2d params:{'padding': '[0, 0, 0, 0]', 'channels': '2048', 'kernel_size': '[1, 1]'}
        #     tops:['call_204'] bottoms:['resnetv24_stage4_conv6_weight', 'call_203']
        #     '''
        #     if layer is not None:
        #         layer.print_self()
        # print("------------------")
        # print("replace_map", self.replace_map)
        # print("------------------")

        dir_path = os.path.abspath(os.path.dirname(relay_python_path))
        print(dir_path)
        if not os.path.exists(dir_path):
            # os.mkdir(dir_path)
            os.makedirs(dir_path)

        # create relay_python.py
        type_transform_map = {
            "dyn.reshape": "reshape",
            "dyn.strided_slice": "strided_slice"
        }
        type_param_ignore_map = {
            "dyn.reshape": ["newshape"],
            "dyn.strided_slice": ["begin", "end", "strides"]
        }
        param_ignore_list = ["out_dtype"]

        with open(relay_python_path, 'w') as relay_python:
            # import
            relay_python.write("import tvm\r")
            relay_python.write("from tvm import relay, IRModule\r")
            relay_python.write("import numpy as np\r\r")
            # def func
            relay_python.write("def {}():\r".format(module_name))

            # input & params
            # print(self.net_inputs)
            # print(self.net_input_shapes)
            for index, net_input in enumerate(self.net_inputs):
                relay_python.write(
                    "    {} = relay.var(\"{}\", shape=(".format(net_input.replace('.', '_').replace("/", "_").replace("::", "_"), net_input))
                # for dim in self.net_input_shapes[index]:
                # relay_python.write("{}, ".format(
                # "relay.Any()" if dim == '?' else dim))
                if len(self.net_input_shapes[index]) == 1:
                    shape = "{}, ".format(self.net_input_shapes[index][0]).replace(
                        "?", "relay.Any()")
                else:
                    shape = ", ".join(self.net_input_shapes[
                        index]).replace("?", "relay.Any()")
                relay_python.write(shape)
                relay_python.write("), dtype=\"float32\")\r")  # TODO:type

            relay_python.write("\r")

            topKey = set()
            for l in self.layer_list:
                if l == None:
                    continue

                for i, bottom in enumerate(l.bottoms):
                    for b in bottom if isinstance(bottom, list) else [bottom]:
                        if "Constant" in b:
                            if "int" in self.net_meta_dtypes[b]:
                                relay_python.write("    {} = relay.const(np.random.randint(0, 10, ({})), dtype=\"{}\")\r".format(
                                    b, self.net_meta_shapes[b].strip('(').strip(')'), self.net_meta_dtypes[b]))
                            else:
                                if len(self.net_meta_shapes[b]) > 0:
                                    relay_python.write("    {} = relay.const(np.random.rand({}), dtype=\"{}\")\r".format(
                                        b, self.net_meta_shapes[b].strip('(').strip(')'), self.net_meta_dtypes[b]))
                                # TODO:
                                else:
                                    relay_python.write("    {} = relay.const(np.random.rand({}), dtype=\"{}\")\r".format(
                                        b, self.net_meta_shapes[b].strip('(').strip(')'), self.net_meta_dtypes[b]))
                        # if b.isdigit() == True:
                        #     relay_python.write("{} = relay.const(np.array({}, dtype=\"float32\"))\r".format("scale_"+b, b))# TODO:type

                # left of =
                for i, top in enumerate(l.tops):
                    topKey.add(top)
                    relay_python.write("    {}".format(top.replace('.', '_')))
                    if i != len(l.tops) - 1:
                        relay_python.write(", ")

                # = & nn.{}
                relay_python.write(" = relay.{}(".format(
                    type_transform_map[l.type] if l.type in type_transform_map else l.type))

                # nn.{content,...params}——content
                # print(l.bottoms)
                # print(l.line)
                flag = False
                if re.match(r"(%\d+, %\d+)", l.line):
                    flag = True
                if isinstance(l.bottoms[0], list):
                    times = 0
                    for _, item in enumerate(l.bottoms[0]):
                        times += 1
                    if times > 1:
                        flag = True

                print("bottoms:", l.bottoms)
                for i, bottom in enumerate(l.bottoms):
                    # call_9 = relay.nn.relu(relay.Tuple([call_7_0]), )
                    # if isinstance(bottom, list):
                    # relay_python.write("relay.Tuple([")
                    # TODO: ? %32 = (%28, %29, %30, %31)
                    # bottoms: [['call_28', 'call_29', 'call_30', 'call_31']]
                    if flag:
                        relay_python.write("relay.Tuple([")
                        # relay_python.write("[")

                    for j, b in enumerate(bottom if isinstance(bottom, list) else [bottom]):
                        # meta[constant]
                        if b.isdigit() == True:
                            relay_python.write(
                                "relay.const(np.array({}, dtype=\"int64\"))".format(b))  # TODO:type
                            # relay_python.write("{}".format("scale_"+b))
                        # fix : %226 = multiply(1f, %resnetv24_dense0_bias);
                        if re.match(r"\d+f", b):
                            relay_python.write(
                                "relay.const({}.0, dtype=\"float32\")".format(b.strip("f")))

                        else:
                            # 'call_201.0' 'call_201_0'
                            # fix split call_205_0(with input call_205)
                            if b not in topKey and ".0" in b:
                                b = b.replace(".0", "")

                            relay_python.write(
                                "{}".format(b.replace('.', '_').replace("/", "_")))
                            # fix: relay.nn.batch_norm —— ValueError: don't know how to convert type <class 'tvm.relay.expr.TupleWrapper'> to object
                            if "call_" in b:
                                # call_num = b.strip("call_").split(".")[0]
                                for idx, layer in enumerate(self.layer_list):
                                    if layer == None:
                                        continue
                                    if layer.tops[0] == b:
                                        if self.layer_list[idx].type == "nn.batch_norm":
                                            relay_python.write("[0]")
                                        break

                        if j != len(bottom if isinstance(bottom, list) else [bottom]) - 1:
                            relay_python.write(", ")

                        # if isinstance(bottom, list):
                        # relay_python.write("])")

                    if flag:
                        relay_python.write("])")
                        # relay_python.write("]")

                    if i != len(l.bottoms) - 1:
                        relay_python.write(", ")

                if (len(l.params) > 0):
                    relay_python.write(", ")

                # nn.{content,...params}——params
                for i, key in enumerate(l.params):
                    # ignore some param
                    if l.type in type_param_ignore_map:
                        if key in type_param_ignore_map[l.type]:
                            continue
                    if key in param_ignore_list:
                        continue
                    relay_python.write("{}={}".format(key, l.params[key]))
                    if i != len(l.params) - 1:
                        relay_python.write(", ")
                relay_python.write(")\n")

            # return & output
            relay_python.write("    return ")
            for i in range(self.output_count):
                relay_python.write("call_output{}".format(i))
                if i != self.output_count - 1:
                    relay_python.write(", ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        "-i",
        type=str,
        help="The path of the IRModule text file",
    )
    args = parser.parse_args()
    relay_text_path = args.input_path
    relay_python_path = relay_text_path.replace(".txt", ".py")

    parser = MyParser()


if __name__ == "__main__":
    main()
