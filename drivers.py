import tvm

class CPU:
    target= "llvm"
    device=tvm.cpu(0)

class GPU:
    target= "cuda"
    device=tvm.gpu(0)

