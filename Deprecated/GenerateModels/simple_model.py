import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from CheckData import CheckData

# 定义模型导出位置
onnx_name="simple-model-fish"                                  # (4,3,14,14) ===> (1,)
onnx_fold=os.path.join(Config.OnnxSaveFold,onnx_name)
os.makedirs(onnx_fold,exist_ok=True)

# 模型定义
input_info={
    "name": "input_edge",
    "shape": (1,15)
}

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self,x):                            
        return F.relu(x)
        # return output

def main():
    model=Model()
    model.eval()        # 设置为推理模型
    
    x = torch.rand(*input_info["shape"],requires_grad=True)
    _ = model(x)          # 计算一次前向传播，https://blog.csdn.net/qq_44930937/article/details/109701307


    torch.onnx.export(model,x,Config.ModelSavePathName(onnx_name),export_params=True,input_names=[input_info["name"]],output_names=["output"])  # "edge"使得自定义名称与tvm内部自动命名显示区分，便于理解

if __name__=="__main__":
    main()

