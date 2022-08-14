import os
import json
import onnx
import onnxruntime
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from GenerateModels.tvm_model import get_tvm_model
import tvm
import onnxruntime as ort
from config import Config
import time
import drivers

input_shape = (1, 3, 224, 224)
test_count = 5
mydriver=drivers.CPU()

def main():

    # onnx model
    onnx_path = Config.ModelSavePathName("resnet50-v2-7")
    # (N,3,224,224)——need to set input size for tvm model
    x = torch.rand(*input_shape, requires_grad=True)
    input_name = "data"
    shape_dict = {input_name: x.shape}
    module, _ = get_tvm_model(

    onnx_path, shape_dict, target=mydriver.target, dev=mydriver.device)

    # write check data to disk
    for _ in range(test_count):
        data_input = torch.rand(*input_shape)
        # onnx model
        sess = ort.InferenceSession(onnx_path)
        print("input name:", sess.get_inputs()[0])
        input_name = sess.get_inputs()[0].name
        onnx_out = sess.run(
            [], {input_name: data_input.numpy()})
        onnx_out = numpy.array(onnx_out).flatten()
        print("onnx %s" % onnx_out[:10])

        # tvm model
        module.set_input(input_name, data_input.numpy())
        start = time.time()
        module.run()
        print("time=", time.time()-start)
        tvm_out = module.get_output(0).numpy().flatten()
        print("tvm %s" % tvm_out[:10])
        print("error: ", (onnx_out[:10]-tvm_out[:10])/tvm_out[:10])


if __name__ == "__main__":
    main()

'''
onnx [-1.1199446   0.8598162   2.2359343   0.795336    0.81137633  0.1695825
  0.5768492  -0.9575536  -0.7314985  -0.848887  ]
time= 0.46112656593322754
tvm [-1.1199456   0.8598145   2.2359312   0.79533494  0.8113759   0.16958097
  0.5768484  -0.957552   -0.7314984  -0.8488851 ]
error:  [-9.5797827e-07  1.9410350e-06  1.3861973e-06  1.3489708e-06
  5.1422836e-07  9.0506592e-06  1.4465933e-06  1.6806663e-06
  8.1482945e-08  2.2468867e-06]
input name: NodeArg(name='data', type='tensor(float)', shape=['N', 3, 224, 224])
onnx [-1.0122778   0.6135392   2.0810874   0.791672    0.99724704  0.3736422
  0.6150685  -0.9172382  -0.8131881  -0.94051623]
time= 0.0017545223236083984
tvm [-1.0122775   0.61353934  2.081086    0.7916714   0.99724764  0.37364405
  0.615071   -0.9172384  -0.8131878  -0.9405156 ]
error:  [ 3.5329035e-07 -1.9429771e-07  6.8738706e-07  7.5289626e-07
 -5.9769150e-07 -4.9451983e-06 -4.0700911e-06 -2.5993086e-07
  3.6648760e-07  6.9711882e-07]
input name: NodeArg(name='data', type='tensor(float)', shape=['N', 3, 224, 224])
onnx [-0.97937745  0.41816142  2.2152832   1.0819664   1.1448352   0.23702678
  0.6751349  -0.8608846  -0.9198257  -0.62278616]
time= 0.0016999244689941406
tvm [-0.97937757  0.4181623   2.2152836   1.0819677   1.1448343   0.23702703
  0.67513543 -0.86088395 -0.9198256  -0.62278634]
error:  [-1.2171944e-07 -2.0668228e-06 -2.1524880e-07 -1.2119606e-06
  8.3302388e-07 -1.0687378e-06 -7.9456919e-07  7.6160217e-07
  6.4799941e-08 -2.8711924e-07]
input name: NodeArg(name='data', type='tensor(float)', shape=['N', 3, 224, 224])
onnx [-1.2354602   0.7694664   2.5134158   0.84798145  0.9970881   0.13786553
  0.52792674 -0.98981494 -1.0078487  -0.7908922 ]
time= 0.0017359256744384766
tvm [-1.2354603   0.7694684   2.5134165   0.84798276  0.99708843  0.13786711
  0.5279277  -0.9898152  -1.0078493  -0.7908934 ]
error:  [-9.6489778e-08 -2.6337116e-06 -2.8457509e-07 -1.5463784e-06
 -3.5867217e-07 -1.1456852e-05 -1.8064487e-06 -2.4087183e-07
 -5.9140433e-07 -1.5072739e-06]
input name: NodeArg(name='data', type='tensor(float)', shape=['N', 3, 224, 224])
onnx [-0.9875667   0.40203175  2.1393046   0.67779505  1.2165266   0.6134354
  0.93964684 -1.1759957  -1.1239022  -1.3757603 ]
time= 0.0017056465148925781
tvm [-0.98756653  0.40203115  2.1393032   0.67779446  1.216526    0.61343473
  0.9396467  -1.1759948  -1.1239012  -1.3757592 ]
error:  [1.8106520e-07 1.4825877e-06 6.6868103e-07 8.7939117e-07 4.8995781e-07
 1.0688196e-06 1.2686607e-07 8.1095118e-07 8.4853923e-07 7.7984839e-07]
'''
