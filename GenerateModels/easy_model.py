import os
import json
import onnx
import onnxruntime
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import tvm_model
import tvm

# 定义模型导出位置
onnx_name = "easy-model"
onnx_fold = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../onnxs", onnx_name)
os.makedirs(onnx_fold, exist_ok=True)

# 模型定义
input_shape = (4, 3, 14, 14)

random_add1 = torch.rand(4, 3, 14, 14)
random_add2 = torch.rand(4, 1, 6, 6)

'''
x -> conv1 ->y1
x ->add1 -> conv2 ->y2

y1+y2 -> y

y -> relu -> z1
y -> add2 -> z2

z1+z2 -> output
'''


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=1, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=3, out_channels=1, kernel_size=4, stride=2)
        self.linear = nn.Linear(in_features=4*1*6*6, out_features=1)

    def forward(self, x):                            # x: (4,3,14,14)        1, 11
        y1 = F.relu(self.conv1(x))                    # y1: (4,1,6,6)
        y2 = F.relu(self.conv2(x+random_add1))        # y2: (4,1,6,6)

        y = y1+y2

        output = F.relu(y)+y+random_add2

        return self.linear(torch.flatten(output, 0))

# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main():
    test_count = 10
    model = Model()
    model.eval()        # 设置为推理模型

    x = torch.rand(*input_shape, requires_grad=True)
    # 计算一次前向传播，https://blog.csdn.net/qq_44930937/article/details/109701307
    _ = model(x)

    # onnx model
    onnx_path = os.path.join(onnx_fold, onnx_name+".onnx")
    input_name = "input"
    output_name = "output"
    torch.onnx.export(model, x, onnx_path, input_names=[
                      input_name], output_names=[output_name])

    # tvm model
    shape_dict = {input_name: x.detach().numpy().shape}
    module = tvm_model.get_tvm_model(
        onnx_path, shape_dict, target="cuda", dev=tvm.cuda())

    # write check data to disk
    datas = {}
    for _ in range(test_count):
        data_input = torch.rand(*input_shape)
        # input-output map
        out = {}
        torch_out = model(data_input).tolist()[0]
        print("torch %s" % torch_out)
        out["pytorch"] = torch_out

        module.set_input(input_name, data_input.numpy())
        module.run()
        tvm_out = module.get_output(0).numpy().tolist()[0]
        print("tvm %s" % tvm_out)
        out["tvm"] = tvm_out
        datas[str(data_input.tolist())] = out

    with open(os.path.join(onnx_fold, "data.json"), "w", encoding="utf-8") as fp:
        fp.write(json.dumps(datas))

    print("datas in:", onnx_fold)


if __name__ == "__main__":
    main()


'''
compare torch & tvm

torch 0.6757051944732666
tvm 0.6757051944732666
torch 0.7329940795898438
tvm 0.7329941987991333
torch 0.8598699569702148
tvm 0.8598699569702148
torch 0.7072029113769531
tvm 0.7072029113769531
torch 0.742712140083313
tvm 0.742712140083313
torch 0.9915422201156616
tvm 0.9915421009063721
torch 0.9976719617843628
tvm 0.9976718425750732
torch 0.660358190536499
tvm 0.6603580713272095
torch 0.5278353691101074
tvm 0.527835488319397
torch 0.5978937149047852
tvm 0.5978938341140747
'''
