# this file is create by program.
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    call_7 = relay.nn.conv2d(call_6, features_3_expand1x1_weight, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_8 = relay.nn.bias_add(call_7, features_3_expand1x1_bias)
    call_9 = relay.nn.conv2d(call_6, features_3_expand3x3_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_10 = relay.nn.bias_add(call_9, features_3_expand3x3_bias)
    call_11 = relay.nn.relu(call_8)
    call_12 = relay.nn.relu(call_10)
    call_output0 = relay.concatenate(relay.Tuple([call_11, call_12]), axis=1)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    call_18 = relay.nn.conv2d(call_17, features_4_expand1x1_weight, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_19 = relay.nn.bias_add(call_18, features_4_expand1x1_bias)
    call_20 = relay.nn.conv2d(call_17, features_4_expand3x3_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_21 = relay.nn.bias_add(call_20, features_4_expand3x3_bias)
    call_22 = relay.nn.relu(call_19)
    call_23 = relay.nn.relu(call_21)
    call_output0 = relay.concatenate(relay.Tuple([call_22, call_23]), axis=1)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    call_29 = relay.nn.conv2d(call_28, features_5_expand1x1_weight, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_30 = relay.nn.bias_add(call_29, features_5_expand1x1_bias)
    call_31 = relay.nn.conv2d(call_28, features_5_expand3x3_weight, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_32 = relay.nn.bias_add(call_31, features_5_expand3x3_bias)
    call_33 = relay.nn.relu(call_30)
    call_34 = relay.nn.relu(call_32)
    call_output0 = relay.concatenate(relay.Tuple([call_33, call_34]), axis=1)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    call_41 = relay.nn.conv2d(call_40, features_7_expand1x1_weight, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_42 = relay.nn.bias_add(call_41, features_7_expand1x1_bias)
    call_43 = relay.nn.conv2d(call_40, features_7_expand3x3_weight, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_44 = relay.nn.bias_add(call_43, features_7_expand3x3_bias)
    call_45 = relay.nn.relu(call_42)
    call_46 = relay.nn.relu(call_44)
    call_output0 = relay.concatenate(relay.Tuple([call_45, call_46]), axis=1)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    call_52 = relay.nn.conv2d(call_51, features_8_expand1x1_weight, padding=[0, 0, 0, 0], channels=192, kernel_size=[1, 1])
    call_53 = relay.nn.bias_add(call_52, features_8_expand1x1_bias)
    call_54 = relay.nn.conv2d(call_51, features_8_expand3x3_weight, padding=[1, 1, 1, 1], channels=192, kernel_size=[3, 3])
    call_55 = relay.nn.bias_add(call_54, features_8_expand3x3_bias)
    call_56 = relay.nn.relu(call_53)
    call_57 = relay.nn.relu(call_55)
    call_output0 = relay.concatenate(relay.Tuple([call_56, call_57]), axis=1)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    call_63 = relay.nn.conv2d(call_62, features_9_expand1x1_weight, padding=[0, 0, 0, 0], channels=192, kernel_size=[1, 1])
    call_64 = relay.nn.bias_add(call_63, features_9_expand1x1_bias)
    call_65 = relay.nn.conv2d(call_62, features_9_expand3x3_weight, padding=[1, 1, 1, 1], channels=192, kernel_size=[3, 3])
    call_66 = relay.nn.bias_add(call_65, features_9_expand3x3_bias)
    call_67 = relay.nn.relu(call_64)
    call_68 = relay.nn.relu(call_66)
    call_output0 = relay.concatenate(relay.Tuple([call_67, call_68]), axis=1)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    call_74 = relay.nn.conv2d(call_73, features_10_expand1x1_weight, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_75 = relay.nn.bias_add(call_74, features_10_expand1x1_bias)
    call_76 = relay.nn.conv2d(call_73, features_10_expand3x3_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_77 = relay.nn.bias_add(call_76, features_10_expand3x3_bias)
    call_78 = relay.nn.relu(call_75)
    call_79 = relay.nn.relu(call_77)
    call_output0 = relay.concatenate(relay.Tuple([call_78, call_79]), axis=1)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    call_86 = relay.nn.conv2d(call_85, features_12_expand1x1_weight, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_87 = relay.nn.bias_add(call_86, features_12_expand1x1_bias)
    call_88 = relay.nn.conv2d(call_85, features_12_expand3x3_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_89 = relay.nn.bias_add(call_88, features_12_expand3x3_bias)
    call_90 = relay.nn.relu(call_87)
    call_91 = relay.nn.relu(call_89)
    call_output0 = relay.concatenate(relay.Tuple([call_90, call_91]), axis=1)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]