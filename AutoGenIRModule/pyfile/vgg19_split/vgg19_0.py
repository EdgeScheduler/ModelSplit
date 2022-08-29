import tvm
    call_1 = relay.nn.bias_add(call_0, vgg0_conv0_bias)
    call_2 = relay.nn.relu(call_1)
    call_3 = relay.nn.conv2d(call_2, vgg0_conv1_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_4 = relay.nn.bias_add(call_3, vgg0_conv1_bias)
    call_5 = relay.nn.relu(call_4)
    call_output0 = relay.nn.max_pool2d(call_5, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0])
    return call_output0