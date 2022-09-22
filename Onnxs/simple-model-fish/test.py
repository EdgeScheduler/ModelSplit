import onnxruntime
import numpy as np
from memory_profiler import profile
# onnxruntime=1.12.1


### 推理

@profile
def run():
    # CUDAExecutionProvider, CPUExecutionProvider

    opts= onnxruntime.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    

    session = onnxruntime.InferenceSession('Onnxs/simple-model-fish/simple-model-fish.onnx', providers=['CUDAExecutionProvider'],sess_options=opts) 
    input_data = np.array(np.random.randn(1,15)).astype(np.float32)
    

    # session = onnxruntime.InferenceSession('Onnxs/googlenet/googlenet.onnx', providers=['CUDAExecutionProvider'])
    # input_data = np.array(np.random.randn(1,3,224,224)).astype(np.float32)

    # session = onnxruntime.InferenceSession('Onnxs/vgg19/vgg19.onnx', providers=['CPUExecutionProvider'])
    # input_data = np.array(np.random.randn(1,3,224,224)).astype(np.float32)

    # print(session.get_providers())
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name

    for _ in range(1):
        result = session.run([label_name], {input_name: input_data})[0]
        # print(result)

run()