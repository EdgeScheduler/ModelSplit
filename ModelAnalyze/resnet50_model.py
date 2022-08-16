import os
import json
import onnx
import onnxruntime
import numpy as np
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
mydriver = drivers.CPU()


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
        onnx_out = np.array(onnx_out).flatten()
        # print("onnx %s" % onnx_out[:10])

        # tvm model
        module.set_input(input_name, data_input.numpy())
        start = time.time()
        module.run()
        print("time=", time.time()-start)
        tvm_out = module.get_output(0).numpy().flatten()
        # print("tvm %s" % tvm_out[:10])
        # print("error: ", (onnx_out[:10]-tvm_out[:10])/tvm_out[:10])

    print("with time_evaluator")
    ftimer = module.module.time_evaluator(
        "run", mydriver.device, repeat=test_count, min_repeat_ms=500, number=1)
    prof_res = np.array(ftimer().results)  # convert to millisecond
    print("Mean inference time (std dev): %f s (%f s)" %
          (np.mean(prof_res), np.std(prof_res)))


if __name__ == "__main__":
    main()

'''
onnx [-1.179503    0.7060435   2.382137    0.9345787   0.8915695   0.0180721
  0.52511424 -0.71846765 -0.8445105  -0.5531975 ]
[01:34:31] /home/onceas/tvm-back/tvm/src/runtime/threading_backend.cc:217: Warning: more than two frequencies detected!
time= 2.872690439224243
tvm [-1.179505    0.7060419   2.3821375   0.9345786   0.8915709   0.01807265
  0.52511495 -0.71846646 -0.84451026 -0.5531972 ]
error:  [-1.7181427e-06  2.2793627e-06 -2.0017197e-07  1.2755406e-07
 -1.6044842e-06 -3.0403977e-05 -1.3620936e-06  1.6592186e-06
  2.8231580e-07  5.3872873e-07]
input name: NodeArg(name='data', type='tensor(float)', shape=['N', 3, 224, 224])
onnx [-1.003843   0.6332309  2.3235254  0.9343694  1.1785377  0.3231591
  0.6490863 -0.8603263 -0.9550347 -0.9299966]
time= 2.868177890777588
tvm [-1.0038435   0.6332323   2.3235233   0.9343706   1.1785374   0.32315814
  0.6490882  -0.8603267  -0.9550356  -0.92999524]
error:  [-5.9376430e-07 -2.1649350e-06  9.2349717e-07 -1.2758245e-06
  3.0345061e-07  2.9511073e-06 -2.9385046e-06 -4.8496986e-07
 -9.9857459e-07  1.4741009e-06]
input name: NodeArg(name='data', type='tensor(float)', shape=['N', 3, 224, 224])
onnx [-1.0813888   0.87012583  2.4054048   1.0327128   1.2301912   0.7948169
  1.0338835  -1.0552691  -0.7199993  -0.8980377 ]
time= 2.893639087677002
tvm [-1.0813897   0.8701274   2.405404    1.0327127   1.2301917   0.7948185
  1.0338839  -1.0552688  -0.7199976  -0.89803594]
error:  [-7.7165987e-07 -1.7810274e-06  2.9735367e-07  1.1543316e-07
 -3.8761206e-07 -2.0247710e-06 -4.6120957e-07  3.3889742e-07
  2.4007506e-06  1.9247946e-06]
input name: NodeArg(name='data', type='tensor(float)', shape=['N', 3, 224, 224])
onnx [-1.2120851   0.6511546   2.71748     0.9694909   1.1684864   0.45498303
  0.7078821  -0.7932444  -0.8084009  -0.8253119 ]
time= 2.885315179824829
tvm [-1.2120854   0.6511531   2.7174816   0.96949077  1.1684885   0.45498297
  0.707882   -0.79324365 -0.80839986 -0.82531166]
error:  [-1.9670115e-07  2.2884267e-06 -6.1414585e-07  1.2296073e-07
 -1.8363614e-06  1.3100413e-07  1.6840278e-07  9.7682516e-07
  1.3271695e-06  2.8888309e-07]
input name: NodeArg(name='data', type='tensor(float)', shape=['N', 3, 224, 224])
onnx [-1.1147768   0.4126072   2.28551     0.96611893  1.0671105   0.28534696
  0.6073218  -0.9988647  -1.0613092  -1.0867649 ]
time= 2.863893508911133
tvm [-1.1147766   0.41260648  2.2855074   0.9661168   1.0671115   0.28534937
  0.60732514 -0.9988657  -1.0613092  -1.0867654 ]
error:  [ 2.1387117e-07  1.7335059e-06  1.1474932e-06  2.2210227e-06
 -8.9369701e-07 -8.4597632e-06 -5.4960019e-06 -1.0144296e-06
 -0.0000000e+00 -4.3876733e-07]

with time_evaluator
Mean inference time (std dev): 2.862977 s (0.013018 s)
'''
