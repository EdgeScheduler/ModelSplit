import py3nvml.py3nvml as py3nvml
import time

py3nvml.nvmlInit()
max_used = 0
max_util = 0
for i in range(220):
    handle = py3nvml.nvmlDeviceGetHandleByIndex(0)  # 这里的0是GPU id
    meminfo = py3nvml.nvmlDeviceGetMemoryInfo(handle)
    utilization = py3nvml.nvmlDeviceGetUtilizationRates(handle)
    # print(meminfo.total / 1024 / 1024)  # 第1块显卡总的显存大小
    # print(meminfo.used / 1024 / 1024)  # 这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
    # print(meminfo.free / 1024 / 1024)  # 第1块显卡剩余显存大小
    # print("GPU Utilization:", utilization.gpu)
    # print("Memory Utilization:", utilization.memory)
    max_used = max(max_used, meminfo.used/1024/1024)
    max_util = max(max_util, utilization.memory)
    time.sleep(0.2)

print("max_used: {}, max_util: {}".format(str(max_used), str(max_util)))
