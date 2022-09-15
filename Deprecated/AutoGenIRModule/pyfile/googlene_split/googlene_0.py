import tvmfrom tvm import relay, IRModuleimport numpy as npdef GoogleNetModule_0():    conv1_7x7_s2_w_0 = relay.var("conv1/7x7_s2_w_0", shape=(64, 3, 7, 7), dtype="float32")    conv1_7x7_s2_b_0 = relay.var("conv1/7x7_s2_b_0", shape=(64, ), dtype="float32")    conv2_3x3_reduce_w_0 = relay.var("conv2/3x3_reduce_w_0", shape=(64, 64, 1, 1), dtype="float32")    conv2_3x3_reduce_b_0 = relay.var("conv2/3x3_reduce_b_0", shape=(64, ), dtype="float32")    data_0 = relay.var("data_0", shape=(1, 3, 224, 224), dtype="float32")    call_0 = relay.nn.conv2d(data_0, conv1_7x7_s2_w_0, strides=[2, 2], padding=[3, 3, 3, 3], channels=64, kernel_size=[7, 7])
    call_1 = relay.nn.bias_add(call_0, conv1_7x7_s2_b_0)
    call_2 = relay.nn.relu(call_1)
    call_3 = relay.nn.max_pool2d(call_2, pool_size=[3, 3], strides=[2, 2], padding=[0, 0, 2, 2])
    call_4 = relay.nn.lrn(call_3, bias=1)
    call_5 = relay.nn.conv2d(call_4, conv2_3x3_reduce_w_0, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_output0 = relay.nn.bias_add(call_5, conv2_3x3_reduce_b_0)
    return call_output0