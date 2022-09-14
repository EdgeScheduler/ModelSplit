# this file is create by program.def Vgg19Module_0():    import tvm    from tvm import relay, IRModule    import numpy as np    vgg0_conv3_weight = relay.var("vgg0_conv3_weight", shape=(128, 128, 3, 3), dtype="float32")    vgg0_conv2_weight = relay.var("vgg0_conv2_weight", shape=(128, 64, 3, 3), dtype="float32")    vgg0_conv4_bias = relay.var("vgg0_conv4_bias", shape=(256, ), dtype="float32")    data = relay.var("data", shape=(1, 3, 224, 224), dtype="float32")    vgg0_conv4_weight = relay.var("vgg0_conv4_weight", shape=(256, 128, 3, 3), dtype="float32")    vgg0_conv1_weight = relay.var("vgg0_conv1_weight", shape=(64, 64, 3, 3), dtype="float32")    vgg0_conv3_bias = relay.var("vgg0_conv3_bias", shape=(128, ), dtype="float32")    vgg0_conv1_bias = relay.var("vgg0_conv1_bias", shape=(64, ), dtype="float32")    vgg0_conv0_weight = relay.var("vgg0_conv0_weight", shape=(64, 3, 3, 3), dtype="float32")    vgg0_conv0_bias = relay.var("vgg0_conv0_bias", shape=(64, ), dtype="float32")    vgg0_conv2_bias = relay.var("vgg0_conv2_bias", shape=(128, ), dtype="float32")    call_0 = relay.nn.conv2d(data, vgg0_conv0_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_1 = relay.nn.bias_add(call_0, vgg0_conv0_bias)
    call_2 = relay.nn.relu(call_1)
    call_3 = relay.nn.conv2d(call_2, vgg0_conv1_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_4 = relay.nn.bias_add(call_3, vgg0_conv1_bias)
    call_5 = relay.nn.relu(call_4)
    call_6 = relay.nn.max_pool2d(call_5, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0])
    call_7 = relay.nn.conv2d(call_6, vgg0_conv2_weight, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_8 = relay.nn.bias_add(call_7, vgg0_conv2_bias)
    call_9 = relay.nn.relu(call_8)
    call_10 = relay.nn.conv2d(call_9, vgg0_conv3_weight, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_11 = relay.nn.bias_add(call_10, vgg0_conv3_bias)
    call_12 = relay.nn.relu(call_11)
    call_13 = relay.nn.max_pool2d(call_12, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0])
    call_14 = relay.nn.conv2d(call_13, vgg0_conv4_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_15 = relay.nn.bias_add(call_14, vgg0_conv4_bias)
    call_output0 = relay.nn.relu(call_15)
    return call_output0def Vgg19Module_1():    import tvm    from tvm import relay, IRModule    import numpy as np    vgg0_conv6_weight = relay.var("vgg0_conv6_weight", shape=(256, 256, 3, 3), dtype="float32")    vgg0_conv9_bias = relay.var("vgg0_conv9_bias", shape=(512, ), dtype="float32")    vgg0_conv8_weight = relay.var("vgg0_conv8_weight", shape=(512, 256, 3, 3), dtype="float32")    vgg0_conv5_weight = relay.var("vgg0_conv5_weight", shape=(256, 256, 3, 3), dtype="float32")    call_16 = relay.var("call_16", shape=(1, 256, 56, 56), dtype="float32")    vgg0_conv7_bias = relay.var("vgg0_conv7_bias", shape=(256, ), dtype="float32")    vgg0_conv6_bias = relay.var("vgg0_conv6_bias", shape=(256, ), dtype="float32")    vgg0_conv7_weight = relay.var("vgg0_conv7_weight", shape=(256, 256, 3, 3), dtype="float32")    vgg0_conv9_weight = relay.var("vgg0_conv9_weight", shape=(512, 512, 3, 3), dtype="float32")    vgg0_conv8_bias = relay.var("vgg0_conv8_bias", shape=(512, ), dtype="float32")    vgg0_conv5_bias = relay.var("vgg0_conv5_bias", shape=(256, ), dtype="float32")    call_17 = relay.nn.conv2d(call_16, vgg0_conv5_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_18 = relay.nn.bias_add(call_17, vgg0_conv5_bias)
    call_19 = relay.nn.relu(call_18)
    call_20 = relay.nn.conv2d(call_19, vgg0_conv6_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_21 = relay.nn.bias_add(call_20, vgg0_conv6_bias)
    call_22 = relay.nn.relu(call_21)
    call_23 = relay.nn.conv2d(call_22, vgg0_conv7_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_24 = relay.nn.bias_add(call_23, vgg0_conv7_bias)
    call_25 = relay.nn.relu(call_24)
    call_26 = relay.nn.max_pool2d(call_25, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0])
    call_27 = relay.nn.conv2d(call_26, vgg0_conv8_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_28 = relay.nn.bias_add(call_27, vgg0_conv8_bias)
    call_29 = relay.nn.relu(call_28)
    call_30 = relay.nn.conv2d(call_29, vgg0_conv9_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_output0 = relay.nn.bias_add(call_30, vgg0_conv9_bias)
    return call_output0def Vgg19Module_2():    import tvm    from tvm import relay, IRModule    import numpy as np    vgg0_conv10_bias = relay.var("vgg0_conv10_bias", shape=(512, ), dtype="float32")    vgg0_conv13_weight = relay.var("vgg0_conv13_weight", shape=(512, 512, 3, 3), dtype="float32")    call_31 = relay.var("call_31", shape=(1, 512, 28, 28), dtype="float32")    vgg0_conv10_weight = relay.var("vgg0_conv10_weight", shape=(512, 512, 3, 3), dtype="float32")    vgg0_conv13_bias = relay.var("vgg0_conv13_bias", shape=(512, ), dtype="float32")    vgg0_conv11_bias = relay.var("vgg0_conv11_bias", shape=(512, ), dtype="float32")    vgg0_conv11_weight = relay.var("vgg0_conv11_weight", shape=(512, 512, 3, 3), dtype="float32")    vgg0_conv12_weight = relay.var("vgg0_conv12_weight", shape=(512, 512, 3, 3), dtype="float32")    vgg0_conv12_bias = relay.var("vgg0_conv12_bias", shape=(512, ), dtype="float32")    vgg0_conv14_weight = relay.var("vgg0_conv14_weight", shape=(512, 512, 3, 3), dtype="float32")    call_32 = relay.nn.relu(call_31)
    call_33 = relay.nn.conv2d(call_32, vgg0_conv10_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_34 = relay.nn.bias_add(call_33, vgg0_conv10_bias)
    call_35 = relay.nn.relu(call_34)
    call_36 = relay.nn.conv2d(call_35, vgg0_conv11_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_37 = relay.nn.bias_add(call_36, vgg0_conv11_bias)
    call_38 = relay.nn.relu(call_37)
    call_39 = relay.nn.max_pool2d(call_38, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0])
    call_40 = relay.nn.conv2d(call_39, vgg0_conv12_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_41 = relay.nn.bias_add(call_40, vgg0_conv12_bias)
    call_42 = relay.nn.relu(call_41)
    call_43 = relay.nn.conv2d(call_42, vgg0_conv13_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_44 = relay.nn.bias_add(call_43, vgg0_conv13_bias)
    call_45 = relay.nn.relu(call_44)
    call_output0 = relay.nn.conv2d(call_45, vgg0_conv14_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    return call_output0def Vgg19Module_3():    import tvm    from tvm import relay, IRModule    import numpy as np    vgg0_dense0_bias = relay.var("vgg0_dense0_bias", shape=(4096, ), dtype="float32")    vgg0_conv14_bias = relay.var("vgg0_conv14_bias", shape=(512, ), dtype="float32")    call_46 = relay.var("call_46", shape=(1, 512, 14, 14), dtype="float32")    vgg0_dense2_bias = relay.var("vgg0_dense2_bias", shape=(1000, ), dtype="float32")    vgg0_dense1_bias = relay.var("vgg0_dense1_bias", shape=(4096, ), dtype="float32")    vgg0_dense1_weight = relay.var("vgg0_dense1_weight", shape=(4096, 4096), dtype="float32")    vgg0_conv15_bias = relay.var("vgg0_conv15_bias", shape=(512, ), dtype="float32")    vgg0_conv15_weight = relay.var("vgg0_conv15_weight", shape=(512, 512, 3, 3), dtype="float32")    vgg0_dense2_weight = relay.var("vgg0_dense2_weight", shape=(1000, 4096), dtype="float32")    vgg0_dense0_weight = relay.var("vgg0_dense0_weight", shape=(4096, 25088), dtype="float32")    call_47 = relay.nn.bias_add(call_46, vgg0_conv14_bias)
    call_48 = relay.nn.relu(call_47)
    call_49 = relay.nn.conv2d(call_48, vgg0_conv15_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_50 = relay.nn.bias_add(call_49, vgg0_conv15_bias)
    call_51 = relay.nn.relu(call_50)
    call_52 = relay.nn.max_pool2d(call_51, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0])
    call_53 = relay.nn.batch_flatten(call_52)
    call_54 = relay.nn.batch_flatten(call_53)
    call_55 = relay.nn.dense(call_54, vgg0_dense0_weight, units=4096)
    call_56 = relay.multiply(relay.const(1.0, dtype="float32"), vgg0_dense0_bias)
    call_57 = relay.add(call_55, call_56)
    call_58 = relay.nn.relu(call_57)
    call_59_0 = relay.nn.dropout(call_58)
    call_61 = relay.nn.batch_flatten(call_59_0)
    call_62 = relay.nn.batch_flatten(call_61)
    call_63 = relay.nn.dense(call_62, vgg0_dense1_weight, units=4096)
    call_64 = relay.multiply(relay.const(1.0, dtype="float32"), vgg0_dense1_bias)
    call_65 = relay.add(call_63, call_64)
    call_66 = relay.nn.relu(call_65)
    call_67_0 = relay.nn.dropout(call_66)
    call_69 = relay.nn.batch_flatten(call_67_0)
    call_70 = relay.nn.batch_flatten(call_69)
    call_71 = relay.nn.dense(call_70, vgg0_dense2_weight, units=1000)
    call_72 = relay.multiply(relay.const(1.0, dtype="float32"), vgg0_dense2_bias)
    call_output0 = relay.add(call_71, call_72)
    return call_output0