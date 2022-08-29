import tvm
    call_1_0 = relay.nn.batch_norm(call_0, models_0_bn1_weight, models_0_bn1_bias, models_0_bn1_running_mean, models_0_bn1_running_var)
    call_3 = relay.nn.leaky_relu(call_1_0[0], alpha=0.1)
    call_4 = relay.nn.max_pool2d(call_3, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0])
    call_5 = relay.nn.conv2d(call_4, models_2_conv2_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_6_0 = relay.nn.batch_norm(call_5, models_2_bn2_weight, models_2_bn2_bias, models_2_bn2_running_mean, models_2_bn2_running_var)
    call_8 = relay.nn.leaky_relu(call_6_0[0], alpha=0.1)
    call_9 = relay.nn.max_pool2d(call_8, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0])
    call_10 = relay.nn.conv2d(call_9, models_4_conv3_weight, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_11_0 = relay.nn.batch_norm(call_10, models_4_bn3_weight, models_4_bn3_bias, models_4_bn3_running_mean, models_4_bn3_running_var)
    call_13 = relay.nn.leaky_relu(call_11_0[0], alpha=0.1)
    call_output0 = relay.nn.conv2d(call_13, models_5_conv4_weight, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    return call_output0