import tvm
    call_8 = relay.nn.conv2d(call_7, conv2_3x3_w_0, padding=[1, 1, 1, 1], channels=192, kernel_size=[3, 3])
    call_9 = relay.nn.bias_add(call_8, conv2_3x3_b_0)
    call_output0 = relay.nn.relu(call_9)
    return call_output0