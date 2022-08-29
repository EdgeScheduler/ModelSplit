import tvm
    call_8 = relay.nn.bias_add(call_7, vgg0_conv2_bias)
    call_9 = relay.nn.relu(call_8)
    call_output0 = relay.nn.conv2d(call_9, vgg0_conv3_weight, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    return call_output0