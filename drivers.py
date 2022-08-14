import tvm

class CPU:
    def __init__(self):
        self.target=CPU.target
        self.device=CPU.device
    target= "llvm"
    device=tvm.cpu(0)

class GPU:
    def __init__(self):
        self.target=GPU.target
        self.device=GPU.device
    target= "cuda"
    device=tvm.cuda(0)
