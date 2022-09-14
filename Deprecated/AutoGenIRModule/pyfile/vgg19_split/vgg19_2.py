import tvm
    call_12 = relay.nn.relu(call_11)
    call_13 = relay.nn.max_pool2d(call_12, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0])
    call_14 = relay.nn.conv2d(call_13, vgg0_conv4_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_output0 = relay.nn.bias_add(call_14, vgg0_conv4_bias)
    return call_output0