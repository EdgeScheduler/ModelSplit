import tvm
    call_101 = relay.nn.leaky_relu(call_99_0[0], alpha=0.1)
    call_102 = relay.nn.conv2d(call_101, models_30_conv23_weight, padding=[0, 0, 0, 0], channels=425, kernel_size=[1, 1])
    call_output0 = relay.nn.bias_add(call_102, models_30_conv23_bias)
    return call_output0