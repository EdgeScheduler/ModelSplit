import tvmfrom tvm import relay, IRModuleimport numpy as npdef GoogleNetModule_1():    conv2_3x3_b_0 = relay.var("conv2/3x3_b_0", shape=(192, ), dtype="float32")    call_6 = relay.var("call_6", shape=(1, 64, 56, 56), dtype="float32")    conv2_3x3_w_0 = relay.var("conv2/3x3_w_0", shape=(192, 64, 3, 3), dtype="float32")    call_7 = relay.nn.relu(call_6)
    call_8 = relay.nn.conv2d(call_7, conv2_3x3_w_0, padding=[1, 1, 1, 1], channels=192, kernel_size=[3, 3])
    call_9 = relay.nn.bias_add(call_8, conv2_3x3_b_0)
    call_output0 = relay.nn.relu(call_9)
    return call_output0