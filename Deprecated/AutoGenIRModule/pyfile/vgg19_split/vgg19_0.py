import tvmfrom tvm import relay, IRModuleimport numpy as npdef Vgg19Module_0():    vgg0_conv1_weight = relay.var("vgg0_conv1_weight", shape=(64, 64, 3, 3), dtype="float32")    data = relay.var("data", shape=(1, 3, 224, 224), dtype="float32")    vgg0_conv1_bias = relay.var("vgg0_conv1_bias", shape=(64, ), dtype="float32")    vgg0_conv0_bias = relay.var("vgg0_conv0_bias", shape=(64, ), dtype="float32")    vgg0_conv0_weight = relay.var("vgg0_conv0_weight", shape=(64, 3, 3, 3), dtype="float32")    call_0 = relay.nn.conv2d(data, vgg0_conv0_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_1 = relay.nn.bias_add(call_0, vgg0_conv0_bias)
    call_2 = relay.nn.relu(call_1)
    call_3 = relay.nn.conv2d(call_2, vgg0_conv1_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_4 = relay.nn.bias_add(call_3, vgg0_conv1_bias)
    call_5 = relay.nn.relu(call_4)
    call_output0 = relay.nn.max_pool2d(call_5, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0])
    return call_output0