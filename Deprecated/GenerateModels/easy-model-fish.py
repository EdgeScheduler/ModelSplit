import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from CheckData import CheckData

# 定义模型导出位置
onnx_name="easy-model-fish"                                  # (4,3,14,14) ===> (1,)
onnx_fold=os.path.join(Config.OnnxSaveFold,onnx_name)
os.makedirs(onnx_fold,exist_ok=True)

# 模型定义
input_info={
    "name": "input_edge",
    "shape": (4,3,14,14)
}

random_add1=torch.rand(4,3,14,14)
random_add2=torch.rand(4,1,6,6)

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
        self.conv1=nn.Conv2d(in_channels=3,out_channels=1,kernel_size=4,stride=2)
        self.conv2=nn.Conv2d(in_channels=3,out_channels=1,kernel_size=4,stride=2)
        self.linear=nn.Linear(in_features=4*1*6*6,out_features=1)
        
    def forward(self,x):                            # x: (4,3,14,14)        1, 11 
        y1=F.relu(self.conv1(x))                    # y1: (4,1,6,6)
        y2=F.relu(self.conv2(x+random_add1))        # y2: (4,1,6,6)

        y=y1+y2

        output=F.relu(y)+y+random_add2

        return F.relu(self.linear(torch.reshape(output,(1,4*1*6*6))))
        # return output

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):                            # x: (4,3,14,14)        1, 11
        y = F.relu(self.conv1(x))                    # y1: (4,1,6,6)
      
        return y

def main():
    model=Model()
    model.eval()        # 设置为推理模型
    
    x = torch.rand(*input_info["shape"],requires_grad=True)
    _ = model(x)          # 计算一次前向传播，https://blog.csdn.net/qq_44930937/article/details/109701307


    torch.onnx.export(model,x,Config.ModelSavePathName(onnx_name),export_params=True,input_names=[input_info["name"]],output_names=["output"])  # "edge"使得自定义名称与tvm内部自动命名显示区分，便于理解

    # write check data to disk
    datas={}
    for _ in range(Config.TestDataCount):
        data_input=torch.rand(*input_info["shape"])
        datas[str(data_input.tolist())]=model(data_input).tolist()[0][0]


    with open(Config.ModelSaveDataPathName(onnx_name),"w",encoding="utf-8") as fp:
        fp.write(json.dumps(datas))

    if CheckData(input_info,onnx_name):
        print("check ok, save to:",onnx_fold)
    else:
        print("validate return bad.")

if __name__=="__main__":
    main()

