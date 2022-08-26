import tvm
    call_1 = relay.nn.bias_add(call_0, bias1)
    call_2 = relay.add(input, add1)
    call_3 = relay.nn.conv2d(call_2, weight2, strides=[2, 2], padding=[0, 0, 0, 0], channels=1, kernel_size=[4, 4])
    call_4 = relay.nn.bias_add(call_3, bias2)
    call_5 = relay.nn.relu(call_1)
    call_6 = relay.nn.relu(call_4)
    call_7 = relay.add(call_5, call_6)
    call_8 = relay.nn.relu(call_7)
    call_output0 = relay.add(call_7, call_8)
    return call_output0