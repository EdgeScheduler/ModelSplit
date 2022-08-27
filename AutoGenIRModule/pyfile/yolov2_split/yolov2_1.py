import tvmfrom tvm import relay, IRModuleimport numpy as npdef YoloModule_1():    models_13_bn10_bias = relay.var("models.13.bn10.bias", shape=(256, ), dtype="float32")    models_14_bn11_running_var = relay.var("models.14.bn11.running_var", shape=(512, ), dtype="float32")    models_12_bn9_running_var = relay.var("models.12.bn9.running_var", shape=(512, ), dtype="float32")    models_5_bn4_bias = relay.var("models.5.bn4.bias", shape=(64, ), dtype="float32")    models_8_bn6_running_var = relay.var("models.8.bn6.running_var", shape=(256, ), dtype="float32")    models_14_bn11_running_mean = relay.var("models.14.bn11.running_mean", shape=(512, ), dtype="float32")    models_9_conv7_weight = relay.var("models.9.conv7.weight", shape=(128, 256, 1, 1), dtype="float32")    models_14_bn11_weight = relay.var("models.14.bn11.weight", shape=(512, ), dtype="float32")    models_6_conv5_weight = relay.var("models.6.conv5.weight", shape=(128, 64, 3, 3), dtype="float32")    models_13_conv10_weight = relay.var("models.13.conv10.weight", shape=(256, 512, 1, 1), dtype="float32")    models_13_bn10_running_var = relay.var("models.13.bn10.running_var", shape=(256, ), dtype="float32")    models_12_conv9_weight = relay.var("models.12.conv9.weight", shape=(512, 256, 3, 3), dtype="float32")    models_10_bn8_running_mean = relay.var("models.10.bn8.running_mean", shape=(256, ), dtype="float32")    models_12_bn9_running_mean = relay.var("models.12.bn9.running_mean", shape=(512, ), dtype="float32")    models_12_bn9_weight = relay.var("models.12.bn9.weight", shape=(512, ), dtype="float32")    models_14_conv11_weight = relay.var("models.14.conv11.weight", shape=(512, 256, 3, 3), dtype="float32")    models_8_conv6_weight = relay.var("models.8.conv6.weight", shape=(256, 128, 3, 3), dtype="float32")    models_9_bn7_running_var = relay.var("models.9.bn7.running_var", shape=(128, ), dtype="float32")    models_8_bn6_bias = relay.var("models.8.bn6.bias", shape=(256, ), dtype="float32")    models_6_bn5_running_mean = relay.var("models.6.bn5.running_mean", shape=(128, ), dtype="float32")    models_10_bn8_running_var = relay.var("models.10.bn8.running_var", shape=(256, ), dtype="float32")    models_5_bn4_running_mean = relay.var("models.5.bn4.running_mean", shape=(64, ), dtype="float32")    models_14_bn11_bias = relay.var("models.14.bn11.bias", shape=(512, ), dtype="float32")    call_14 = relay.var("call_14", shape=(1, 64, 104, 104), dtype="float32")    models_12_bn9_bias = relay.var("models.12.bn9.bias", shape=(512, ), dtype="float32")    models_15_conv12_weight = relay.var("models.15.conv12.weight", shape=(256, 512, 1, 1), dtype="float32")    models_5_bn4_running_var = relay.var("models.5.bn4.running_var", shape=(64, ), dtype="float32")    models_9_bn7_weight = relay.var("models.9.bn7.weight", shape=(128, ), dtype="float32")    models_6_bn5_running_var = relay.var("models.6.bn5.running_var", shape=(128, ), dtype="float32")    models_8_bn6_weight = relay.var("models.8.bn6.weight", shape=(256, ), dtype="float32")    models_13_bn10_weight = relay.var("models.13.bn10.weight", shape=(256, ), dtype="float32")    models_8_bn6_running_mean = relay.var("models.8.bn6.running_mean", shape=(256, ), dtype="float32")    models_6_bn5_weight = relay.var("models.6.bn5.weight", shape=(128, ), dtype="float32")    models_10_bn8_bias = relay.var("models.10.bn8.bias", shape=(256, ), dtype="float32")    models_10_conv8_weight = relay.var("models.10.conv8.weight", shape=(256, 128, 3, 3), dtype="float32")    models_13_bn10_running_mean = relay.var("models.13.bn10.running_mean", shape=(256, ), dtype="float32")    models_5_bn4_weight = relay.var("models.5.bn4.weight", shape=(64, ), dtype="float32")    models_10_bn8_weight = relay.var("models.10.bn8.weight", shape=(256, ), dtype="float32")    models_9_bn7_bias = relay.var("models.9.bn7.bias", shape=(128, ), dtype="float32")    models_6_bn5_bias = relay.var("models.6.bn5.bias", shape=(128, ), dtype="float32")    models_9_bn7_running_mean = relay.var("models.9.bn7.running_mean", shape=(128, ), dtype="float32")    call_15_0 = relay.nn.batch_norm(call_14, models_5_bn4_weight, models_5_bn4_bias, models_5_bn4_running_mean, models_5_bn4_running_var)
    call_17 = relay.nn.leaky_relu(call_15_0[0], alpha=0.1)
    call_18 = relay.nn.conv2d(call_17, models_6_conv5_weight, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_19_0 = relay.nn.batch_norm(call_18, models_6_bn5_weight, models_6_bn5_bias, models_6_bn5_running_mean, models_6_bn5_running_var)
    call_21 = relay.nn.leaky_relu(call_19_0[0], alpha=0.1)
    call_22 = relay.nn.max_pool2d(call_21, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0])
    call_23 = relay.nn.conv2d(call_22, models_8_conv6_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_24_0 = relay.nn.batch_norm(call_23, models_8_bn6_weight, models_8_bn6_bias, models_8_bn6_running_mean, models_8_bn6_running_var)
    call_26 = relay.nn.leaky_relu(call_24_0[0], alpha=0.1)
    call_27 = relay.nn.conv2d(call_26, models_9_conv7_weight, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_28_0 = relay.nn.batch_norm(call_27, models_9_bn7_weight, models_9_bn7_bias, models_9_bn7_running_mean, models_9_bn7_running_var)
    call_30 = relay.nn.leaky_relu(call_28_0[0], alpha=0.1)
    call_31 = relay.nn.conv2d(call_30, models_10_conv8_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_32_0 = relay.nn.batch_norm(call_31, models_10_bn8_weight, models_10_bn8_bias, models_10_bn8_running_mean, models_10_bn8_running_var)
    call_34 = relay.nn.leaky_relu(call_32_0[0], alpha=0.1)
    call_35 = relay.nn.max_pool2d(call_34, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0])
    call_36 = relay.nn.conv2d(call_35, models_12_conv9_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_37_0 = relay.nn.batch_norm(call_36, models_12_bn9_weight, models_12_bn9_bias, models_12_bn9_running_mean, models_12_bn9_running_var)
    call_39 = relay.nn.leaky_relu(call_37_0[0], alpha=0.1)
    call_40 = relay.nn.conv2d(call_39, models_13_conv10_weight, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_41_0 = relay.nn.batch_norm(call_40, models_13_bn10_weight, models_13_bn10_bias, models_13_bn10_running_mean, models_13_bn10_running_var)
    call_43 = relay.nn.leaky_relu(call_41_0[0], alpha=0.1)
    call_44 = relay.nn.conv2d(call_43, models_14_conv11_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_45_0 = relay.nn.batch_norm(call_44, models_14_bn11_weight, models_14_bn11_bias, models_14_bn11_running_mean, models_14_bn11_running_var)
    call_47 = relay.nn.leaky_relu(call_45_0[0], alpha=0.1)
    call_output0 = relay.nn.conv2d(call_47, models_15_conv12_weight, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    return call_output0