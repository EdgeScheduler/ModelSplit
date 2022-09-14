import tvmfrom tvm import relay, IRModuleimport numpy as npdef EasyModule_0():    weight2 = relay.var("weight2", shape=(1, 3, 4, 4), dtype="float32")    add1 = relay.var("add1", shape=(4, 3, 14, 14), dtype="float32")    bias1 = relay.var("bias1", shape=(1, ), dtype="float32")    bias2 = relay.var("bias2", shape=(1, ), dtype="float32")    add2 = relay.var("add2", shape=(4, 1, 6, 6), dtype="float32")    weight1 = relay.var("weight1", shape=(1, 3, 4, 4), dtype="float32")    input = relay.var("input", shape=(4, 3, 14, 14), dtype="float32")    call_0 = relay.nn.conv2d(input, weight1, strides=[2, 2], padding=[0, 0, 0, 0], channels=1, kernel_size=[4, 4])
    call_1 = relay.nn.bias_add(call_0, bias1)
    call_2 = relay.add(input, add1)
    call_3 = relay.nn.conv2d(call_2, weight2, strides=[2, 2], padding=[0, 0, 0, 0], channels=1, kernel_size=[4, 4])
    call_4 = relay.nn.bias_add(call_3, bias2)
    call_5 = relay.nn.relu(call_1)
    call_6 = relay.nn.relu(call_4)
    call_7 = relay.add(call_5, call_6)
    call_8 = relay.nn.relu(call_7)
    call_9 = relay.add(call_7, call_8)
    call_output0 = relay.add(call_9, add2)
    return call_output0