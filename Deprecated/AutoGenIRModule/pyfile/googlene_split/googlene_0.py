import tvm
    call_1 = relay.nn.bias_add(call_0, conv1_7x7_s2_b_0)
    call_2 = relay.nn.relu(call_1)
    call_3 = relay.nn.max_pool2d(call_2, pool_size=[3, 3], strides=[2, 2], padding=[0, 0, 2, 2])
    call_4 = relay.nn.lrn(call_3, bias=1)
    call_5 = relay.nn.conv2d(call_4, conv2_3x3_reduce_w_0, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_output0 = relay.nn.bias_add(call_5, conv2_3x3_reduce_b_0)
    return call_output0