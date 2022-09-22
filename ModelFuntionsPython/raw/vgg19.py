# this file is create by program.def Vgg19(pre_input=None):    import tvm    import numpy as np    from tvm import relay    data = pre_input if pre_input is not None else relay.var("data", shape=(15, 3, 224, 224), dtype="float32")    features_0_weight = relay.var("features.0.weight", shape=(64, 3, 3, 3), dtype="float32")    features_0_bias = relay.var("features.0.bias", shape=(64, ), dtype="float32")    features_2_weight = relay.var("features.2.weight", shape=(64, 64, 3, 3), dtype="float32")    features_2_bias = relay.var("features.2.bias", shape=(64, ), dtype="float32")    features_5_weight = relay.var("features.5.weight", shape=(128, 64, 3, 3), dtype="float32")    features_5_bias = relay.var("features.5.bias", shape=(128, ), dtype="float32")    features_7_weight = relay.var("features.7.weight", shape=(128, 128, 3, 3), dtype="float32")    features_7_bias = relay.var("features.7.bias", shape=(128, ), dtype="float32")    features_10_weight = relay.var("features.10.weight", shape=(256, 128, 3, 3), dtype="float32")    features_10_bias = relay.var("features.10.bias", shape=(256, ), dtype="float32")    features_12_weight = relay.var("features.12.weight", shape=(256, 256, 3, 3), dtype="float32")    features_12_bias = relay.var("features.12.bias", shape=(256, ), dtype="float32")    features_14_weight = relay.var("features.14.weight", shape=(256, 256, 3, 3), dtype="float32")    features_14_bias = relay.var("features.14.bias", shape=(256, ), dtype="float32")    features_16_weight = relay.var("features.16.weight", shape=(256, 256, 3, 3), dtype="float32")    features_16_bias = relay.var("features.16.bias", shape=(256, ), dtype="float32")    features_19_weight = relay.var("features.19.weight", shape=(512, 256, 3, 3), dtype="float32")    features_19_bias = relay.var("features.19.bias", shape=(512, ), dtype="float32")    features_21_weight = relay.var("features.21.weight", shape=(512, 512, 3, 3), dtype="float32")    features_21_bias = relay.var("features.21.bias", shape=(512, ), dtype="float32")    features_23_weight = relay.var("features.23.weight", shape=(512, 512, 3, 3), dtype="float32")    features_23_bias = relay.var("features.23.bias", shape=(512, ), dtype="float32")    features_25_weight = relay.var("features.25.weight", shape=(512, 512, 3, 3), dtype="float32")    features_25_bias = relay.var("features.25.bias", shape=(512, ), dtype="float32")    features_28_weight = relay.var("features.28.weight", shape=(512, 512, 3, 3), dtype="float32")    features_28_bias = relay.var("features.28.bias", shape=(512, ), dtype="float32")    features_30_weight = relay.var("features.30.weight", shape=(512, 512, 3, 3), dtype="float32")    features_30_bias = relay.var("features.30.bias", shape=(512, ), dtype="float32")    features_32_weight = relay.var("features.32.weight", shape=(512, 512, 3, 3), dtype="float32")    features_32_bias = relay.var("features.32.bias", shape=(512, ), dtype="float32")    features_34_weight = relay.var("features.34.weight", shape=(512, 512, 3, 3), dtype="float32")    features_34_bias = relay.var("features.34.bias", shape=(512, ), dtype="float32")    classifier_0_weight = relay.var("classifier.0.weight", shape=(4096, 25088), dtype="float32")    classifier_0_bias = relay.var("classifier.0.bias", shape=(4096, ), dtype="float32")    classifier_3_weight = relay.var("classifier.3.weight", shape=(4096, 4096), dtype="float32")    classifier_3_bias = relay.var("classifier.3.bias", shape=(4096, ), dtype="float32")    classifier_6_weight = relay.var("classifier.6.weight", shape=(1000, 4096), dtype="float32")    classifier_6_bias = relay.var("classifier.6.bias", shape=(1000, ), dtype="float32")    call_0 = relay.nn.conv2d(data, features_0_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_1 = relay.nn.bias_add(call_0, features_0_bias)
    call_2 = relay.nn.relu(call_1)
    call_3 = relay.nn.conv2d(call_2, features_2_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_4 = relay.nn.bias_add(call_3, features_2_bias)
    call_5 = relay.nn.relu(call_4)
    call_6 = relay.nn.max_pool2d(call_5, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0])
    call_7 = relay.nn.conv2d(call_6, features_5_weight, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_8 = relay.nn.bias_add(call_7, features_5_bias)
    call_9 = relay.nn.relu(call_8)
    call_10 = relay.nn.conv2d(call_9, features_7_weight, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_11 = relay.nn.bias_add(call_10, features_7_bias)
    call_12 = relay.nn.relu(call_11)
    call_13 = relay.nn.max_pool2d(call_12, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0])
    call_14 = relay.nn.conv2d(call_13, features_10_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_15 = relay.nn.bias_add(call_14, features_10_bias)
    call_16 = relay.nn.relu(call_15)
    call_17 = relay.nn.conv2d(call_16, features_12_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_18 = relay.nn.bias_add(call_17, features_12_bias)
    call_19 = relay.nn.relu(call_18)
    call_20 = relay.nn.conv2d(call_19, features_14_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_21 = relay.nn.bias_add(call_20, features_14_bias)
    call_22 = relay.nn.relu(call_21)
    call_23 = relay.nn.conv2d(call_22, features_16_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_24 = relay.nn.bias_add(call_23, features_16_bias)
    call_25 = relay.nn.relu(call_24)
    call_26 = relay.nn.max_pool2d(call_25, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0])
    call_27 = relay.nn.conv2d(call_26, features_19_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_28 = relay.nn.bias_add(call_27, features_19_bias)
    call_29 = relay.nn.relu(call_28)
    call_30 = relay.nn.conv2d(call_29, features_21_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_31 = relay.nn.bias_add(call_30, features_21_bias)
    call_32 = relay.nn.relu(call_31)
    call_33 = relay.nn.conv2d(call_32, features_23_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_34 = relay.nn.bias_add(call_33, features_23_bias)
    call_35 = relay.nn.relu(call_34)
    call_36 = relay.nn.conv2d(call_35, features_25_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_37 = relay.nn.bias_add(call_36, features_25_bias)
    call_38 = relay.nn.relu(call_37)
    call_39 = relay.nn.max_pool2d(call_38, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0])
    call_40 = relay.nn.conv2d(call_39, features_28_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_41 = relay.nn.bias_add(call_40, features_28_bias)
    call_42 = relay.nn.relu(call_41)
    call_43 = relay.nn.conv2d(call_42, features_30_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_44 = relay.nn.bias_add(call_43, features_30_bias)
    call_45 = relay.nn.relu(call_44)
    call_46 = relay.nn.conv2d(call_45, features_32_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_47 = relay.nn.bias_add(call_46, features_32_bias)
    call_48 = relay.nn.relu(call_47)
    call_49 = relay.nn.conv2d(call_48, features_34_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_50 = relay.nn.bias_add(call_49, features_34_bias)
    call_51 = relay.nn.relu(call_50)
    call_52 = relay.nn.max_pool2d(call_51, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0])
    call_53 = relay.nn.avg_pool2d(call_52, pool_size=[1, 1], padding=[0, 0, 0, 0])
    call_54 = relay.nn.batch_flatten(call_53)
    call_55 = relay.nn.batch_flatten(call_54)
    call_56 = relay.nn.dense(call_55, classifier_0_weight, units=4096)
    call_57 = relay.multiply(relay.const(1, dtype="float32"), classifier_0_bias)
    call_58 = relay.add(call_56, call_57)
    call_59 = relay.nn.relu(call_58)
    call_60 = relay.nn.batch_flatten(call_59)
    call_61 = relay.nn.dense(call_60, classifier_3_weight, units=4096)
    call_62 = relay.multiply(relay.const(1, dtype="float32"), classifier_3_bias)
    call_63 = relay.add(call_61, call_62)
    call_64 = relay.nn.relu(call_63)
    call_65 = relay.nn.batch_flatten(call_64)
    call_66 = relay.nn.dense(call_65, classifier_6_weight, units=1000)
    call_67 = relay.multiply(relay.const(1, dtype="float32"), classifier_6_bias)
    call_output0 = relay.add(call_66, call_67)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]