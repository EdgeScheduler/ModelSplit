import tvm

class CPU:
    def __init__(self):
        self.target=CPU.target
        self.device=CPU.device
        self.kind=CPU.kind
    target= "llvm"
    device=tvm.cpu(0)
    kind="CPU"

class GPU:
    def __init__(self):
        self.target=GPU.target
        self.device=GPU.device
        self.kind=GPU.kind
    target= "cuda"
    device=tvm.cuda(0)
    kind="GPU"
