# this file is create by program.def SqueezeNetv1_0(pre_input=None):    import tvm    from tvm import relay    data_0 = pre_input if pre_input is not None else relay.var("data_0", shape=(1, 3, 224, 224), dtype="float32")    conv1_b_0 = relay.var("conv1_b_0", shape=(64, ), dtype="float32")    conv1_w_0 = relay.var("conv1_w_0", shape=(64, 3, 3, 3), dtype="float32")    call_0 = relay.nn.conv2d(data_0, conv1_w_0, strides=[2, 2], padding=[0, 0, 0, 0], channels=64, kernel_size=[3, 3])
    call_output0 = relay.nn.bias_add(call_0, conv1_b_0)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_1(pre_input=None):    import tvm    from tvm import relay    call_1 = pre_input if pre_input is not None else relay.var("call_1", shape=(1, 64, 111, 111), dtype="float32")    call_output0 = relay.nn.relu(call_1)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_2(pre_input=None):    import tvm    from tvm import relay    call_2 = pre_input if pre_input is not None else relay.var("call_2", shape=(1, 64, 111, 111), dtype="float32")    call_output0 = relay.nn.max_pool2d(call_2, pool_size=[3, 3], strides=[2, 2], padding=[0, 0, 0, 0])
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_3(pre_input=None):    import tvm    from tvm import relay    call_3 = pre_input if pre_input is not None else relay.var("call_3", shape=(1, 64, 55, 55), dtype="float32")    fire2_squeeze1x1_w_0 = relay.var("fire2/squeeze1x1_w_0", shape=(16, 64, 1, 1), dtype="float32")    call_output0 = relay.nn.conv2d(call_3, fire2_squeeze1x1_w_0, padding=[0, 0, 0, 0], channels=16, kernel_size=[1, 1])
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_4(pre_input=None):    import tvm    from tvm import relay    call_4 = pre_input if pre_input is not None else relay.var("call_4", shape=(1, 16, 55, 55), dtype="float32")    fire2_squeeze1x1_b_0 = relay.var("fire2/squeeze1x1_b_0", shape=(16, ), dtype="float32")    call_output0 = relay.nn.bias_add(call_4, fire2_squeeze1x1_b_0)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_5(pre_input=None):    import tvm    from tvm import relay    fire2_expand3x3_w_0 = relay.var("fire2/expand3x3_w_0", shape=(64, 16, 3, 3), dtype="float32")    fire2_expand1x1_w_0 = relay.var("fire2/expand1x1_w_0", shape=(64, 16, 1, 1), dtype="float32")    fire2_expand1x1_b_0 = relay.var("fire2/expand1x1_b_0", shape=(64, ), dtype="float32")    call_5 = pre_input if pre_input is not None else relay.var("call_5", shape=(1, 16, 55, 55), dtype="float32")    fire2_expand3x3_b_0 = relay.var("fire2/expand3x3_b_0", shape=(64, ), dtype="float32")    call_6 = relay.nn.relu(call_5)
    call_7 = relay.nn.conv2d(call_6, fire2_expand1x1_w_0, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_8 = relay.nn.bias_add(call_7, fire2_expand1x1_b_0)
    call_9 = relay.nn.conv2d(call_6, fire2_expand3x3_w_0, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_10 = relay.nn.bias_add(call_9, fire2_expand3x3_b_0)
    call_11 = relay.nn.relu(call_8)
    call_12 = relay.nn.relu(call_10)
    call_output0 = relay.concatenate(relay.Tuple([call_11, call_12]), axis=1)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_6(pre_input=None):    import tvm    from tvm import relay    call_14 = pre_input if pre_input is not None else relay.var("call_14", shape=(1, 128, 55, 55), dtype="float32")    fire3_squeeze1x1_w_0 = relay.var("fire3/squeeze1x1_w_0", shape=(16, 128, 1, 1), dtype="float32")    call_output0 = relay.nn.conv2d(call_14, fire3_squeeze1x1_w_0, padding=[0, 0, 0, 0], channels=16, kernel_size=[1, 1])
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_7(pre_input=None):    import tvm    from tvm import relay    fire3_squeeze1x1_b_0 = relay.var("fire3/squeeze1x1_b_0", shape=(16, ), dtype="float32")    call_15 = pre_input if pre_input is not None else relay.var("call_15", shape=(1, 16, 55, 55), dtype="float32")    call_output0 = relay.nn.bias_add(call_15, fire3_squeeze1x1_b_0)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_8(pre_input=None):    import tvm    from tvm import relay    call_16 = pre_input if pre_input is not None else relay.var("call_16", shape=(1, 16, 55, 55), dtype="float32")    fire3_expand3x3_b_0 = relay.var("fire3/expand3x3_b_0", shape=(64, ), dtype="float32")    fire3_expand3x3_w_0 = relay.var("fire3/expand3x3_w_0", shape=(64, 16, 3, 3), dtype="float32")    fire3_expand1x1_b_0 = relay.var("fire3/expand1x1_b_0", shape=(64, ), dtype="float32")    fire3_expand1x1_w_0 = relay.var("fire3/expand1x1_w_0", shape=(64, 16, 1, 1), dtype="float32")    call_17 = relay.nn.relu(call_16)
    call_18 = relay.nn.conv2d(call_17, fire3_expand1x1_w_0, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_19 = relay.nn.bias_add(call_18, fire3_expand1x1_b_0)
    call_20 = relay.nn.conv2d(call_17, fire3_expand3x3_w_0, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_21 = relay.nn.bias_add(call_20, fire3_expand3x3_b_0)
    call_22 = relay.nn.relu(call_19)
    call_23 = relay.nn.relu(call_21)
    call_output0 = relay.concatenate(relay.Tuple([call_22, call_23]), axis=1)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_9(pre_input=None):    import tvm    from tvm import relay    call_25 = pre_input if pre_input is not None else relay.var("call_25", shape=(1, 128, 55, 55), dtype="float32")    call_output0 = relay.nn.max_pool2d(call_25, pool_size=[3, 3], strides=[2, 2], padding=[0, 0, 0, 0])
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_10(pre_input=None):    import tvm    from tvm import relay    fire4_squeeze1x1_w_0 = relay.var("fire4/squeeze1x1_w_0", shape=(32, 128, 1, 1), dtype="float32")    call_26 = pre_input if pre_input is not None else relay.var("call_26", shape=(1, 128, 27, 27), dtype="float32")    call_output0 = relay.nn.conv2d(call_26, fire4_squeeze1x1_w_0, padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1])
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_11(pre_input=None):    import tvm    from tvm import relay    fire4_squeeze1x1_b_0 = relay.var("fire4/squeeze1x1_b_0", shape=(32, ), dtype="float32")    call_27 = pre_input if pre_input is not None else relay.var("call_27", shape=(1, 32, 27, 27), dtype="float32")    call_output0 = relay.nn.bias_add(call_27, fire4_squeeze1x1_b_0)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_12(pre_input=None):    import tvm    from tvm import relay    fire4_expand3x3_w_0 = relay.var("fire4/expand3x3_w_0", shape=(128, 32, 3, 3), dtype="float32")    call_28 = pre_input if pre_input is not None else relay.var("call_28", shape=(1, 32, 27, 27), dtype="float32")    fire4_expand1x1_b_0 = relay.var("fire4/expand1x1_b_0", shape=(128, ), dtype="float32")    fire4_expand1x1_w_0 = relay.var("fire4/expand1x1_w_0", shape=(128, 32, 1, 1), dtype="float32")    fire4_expand3x3_b_0 = relay.var("fire4/expand3x3_b_0", shape=(128, ), dtype="float32")    call_29 = relay.nn.relu(call_28)
    call_30 = relay.nn.conv2d(call_29, fire4_expand1x1_w_0, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_31 = relay.nn.bias_add(call_30, fire4_expand1x1_b_0)
    call_32 = relay.nn.conv2d(call_29, fire4_expand3x3_w_0, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_33 = relay.nn.bias_add(call_32, fire4_expand3x3_b_0)
    call_34 = relay.nn.relu(call_31)
    call_35 = relay.nn.relu(call_33)
    call_output0 = relay.concatenate(relay.Tuple([call_34, call_35]), axis=1)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_13(pre_input=None):    import tvm    from tvm import relay    call_37 = pre_input if pre_input is not None else relay.var("call_37", shape=(1, 256, 27, 27), dtype="float32")    fire5_squeeze1x1_w_0 = relay.var("fire5/squeeze1x1_w_0", shape=(32, 256, 1, 1), dtype="float32")    call_output0 = relay.nn.conv2d(call_37, fire5_squeeze1x1_w_0, padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1])
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_14(pre_input=None):    import tvm    from tvm import relay    call_38 = pre_input if pre_input is not None else relay.var("call_38", shape=(1, 32, 27, 27), dtype="float32")    fire5_squeeze1x1_b_0 = relay.var("fire5/squeeze1x1_b_0", shape=(32, ), dtype="float32")    call_output0 = relay.nn.bias_add(call_38, fire5_squeeze1x1_b_0)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_15(pre_input=None):    import tvm    from tvm import relay    fire5_expand1x1_w_0 = relay.var("fire5/expand1x1_w_0", shape=(128, 32, 1, 1), dtype="float32")    call_39 = pre_input if pre_input is not None else relay.var("call_39", shape=(1, 32, 27, 27), dtype="float32")    fire5_expand3x3_b_0 = relay.var("fire5/expand3x3_b_0", shape=(128, ), dtype="float32")    fire5_expand1x1_b_0 = relay.var("fire5/expand1x1_b_0", shape=(128, ), dtype="float32")    fire5_expand3x3_w_0 = relay.var("fire5/expand3x3_w_0", shape=(128, 32, 3, 3), dtype="float32")    call_40 = relay.nn.relu(call_39)
    call_41 = relay.nn.conv2d(call_40, fire5_expand1x1_w_0, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_42 = relay.nn.bias_add(call_41, fire5_expand1x1_b_0)
    call_43 = relay.nn.conv2d(call_40, fire5_expand3x3_w_0, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_44 = relay.nn.bias_add(call_43, fire5_expand3x3_b_0)
    call_45 = relay.nn.relu(call_42)
    call_46 = relay.nn.relu(call_44)
    call_output0 = relay.concatenate(relay.Tuple([call_45, call_46]), axis=1)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_16(pre_input=None):    import tvm    from tvm import relay    call_48 = pre_input if pre_input is not None else relay.var("call_48", shape=(1, 256, 27, 27), dtype="float32")    call_output0 = relay.nn.max_pool2d(call_48, pool_size=[3, 3], strides=[2, 2], padding=[0, 0, 0, 0])
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_17(pre_input=None):    import tvm    from tvm import relay    call_49 = pre_input if pre_input is not None else relay.var("call_49", shape=(1, 256, 13, 13), dtype="float32")    fire6_squeeze1x1_w_0 = relay.var("fire6/squeeze1x1_w_0", shape=(48, 256, 1, 1), dtype="float32")    call_output0 = relay.nn.conv2d(call_49, fire6_squeeze1x1_w_0, padding=[0, 0, 0, 0], channels=48, kernel_size=[1, 1])
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_18(pre_input=None):    import tvm    from tvm import relay    fire6_squeeze1x1_b_0 = relay.var("fire6/squeeze1x1_b_0", shape=(48, ), dtype="float32")    call_50 = pre_input if pre_input is not None else relay.var("call_50", shape=(1, 48, 13, 13), dtype="float32")    call_output0 = relay.nn.bias_add(call_50, fire6_squeeze1x1_b_0)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_19(pre_input=None):    import tvm    from tvm import relay    fire6_expand1x1_b_0 = relay.var("fire6/expand1x1_b_0", shape=(192, ), dtype="float32")    fire6_expand3x3_w_0 = relay.var("fire6/expand3x3_w_0", shape=(192, 48, 3, 3), dtype="float32")    call_51 = pre_input if pre_input is not None else relay.var("call_51", shape=(1, 48, 13, 13), dtype="float32")    fire6_expand3x3_b_0 = relay.var("fire6/expand3x3_b_0", shape=(192, ), dtype="float32")    fire6_expand1x1_w_0 = relay.var("fire6/expand1x1_w_0", shape=(192, 48, 1, 1), dtype="float32")    call_52 = relay.nn.relu(call_51)
    call_53 = relay.nn.conv2d(call_52, fire6_expand1x1_w_0, padding=[0, 0, 0, 0], channels=192, kernel_size=[1, 1])
    call_54 = relay.nn.bias_add(call_53, fire6_expand1x1_b_0)
    call_55 = relay.nn.conv2d(call_52, fire6_expand3x3_w_0, padding=[1, 1, 1, 1], channels=192, kernel_size=[3, 3])
    call_56 = relay.nn.bias_add(call_55, fire6_expand3x3_b_0)
    call_57 = relay.nn.relu(call_54)
    call_58 = relay.nn.relu(call_56)
    call_output0 = relay.concatenate(relay.Tuple([call_57, call_58]), axis=1)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_20(pre_input=None):    import tvm    from tvm import relay    fire7_squeeze1x1_w_0 = relay.var("fire7/squeeze1x1_w_0", shape=(48, 384, 1, 1), dtype="float32")    call_60 = pre_input if pre_input is not None else relay.var("call_60", shape=(1, 384, 13, 13), dtype="float32")    call_output0 = relay.nn.conv2d(call_60, fire7_squeeze1x1_w_0, padding=[0, 0, 0, 0], channels=48, kernel_size=[1, 1])
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_21(pre_input=None):    import tvm    from tvm import relay    call_61 = pre_input if pre_input is not None else relay.var("call_61", shape=(1, 48, 13, 13), dtype="float32")    fire7_squeeze1x1_b_0 = relay.var("fire7/squeeze1x1_b_0", shape=(48, ), dtype="float32")    call_output0 = relay.nn.bias_add(call_61, fire7_squeeze1x1_b_0)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_22(pre_input=None):    import tvm    from tvm import relay    fire7_expand3x3_w_0 = relay.var("fire7/expand3x3_w_0", shape=(192, 48, 3, 3), dtype="float32")    call_62 = pre_input if pre_input is not None else relay.var("call_62", shape=(1, 48, 13, 13), dtype="float32")    fire7_expand3x3_b_0 = relay.var("fire7/expand3x3_b_0", shape=(192, ), dtype="float32")    fire7_expand1x1_b_0 = relay.var("fire7/expand1x1_b_0", shape=(192, ), dtype="float32")    fire7_expand1x1_w_0 = relay.var("fire7/expand1x1_w_0", shape=(192, 48, 1, 1), dtype="float32")    call_63 = relay.nn.relu(call_62)
    call_64 = relay.nn.conv2d(call_63, fire7_expand1x1_w_0, padding=[0, 0, 0, 0], channels=192, kernel_size=[1, 1])
    call_65 = relay.nn.bias_add(call_64, fire7_expand1x1_b_0)
    call_66 = relay.nn.conv2d(call_63, fire7_expand3x3_w_0, padding=[1, 1, 1, 1], channels=192, kernel_size=[3, 3])
    call_67 = relay.nn.bias_add(call_66, fire7_expand3x3_b_0)
    call_68 = relay.nn.relu(call_65)
    call_69 = relay.nn.relu(call_67)
    call_output0 = relay.concatenate(relay.Tuple([call_68, call_69]), axis=1)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_23(pre_input=None):    import tvm    from tvm import relay    call_71 = pre_input if pre_input is not None else relay.var("call_71", shape=(1, 384, 13, 13), dtype="float32")    fire8_squeeze1x1_w_0 = relay.var("fire8/squeeze1x1_w_0", shape=(64, 384, 1, 1), dtype="float32")    call_output0 = relay.nn.conv2d(call_71, fire8_squeeze1x1_w_0, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_24(pre_input=None):    import tvm    from tvm import relay    fire8_squeeze1x1_b_0 = relay.var("fire8/squeeze1x1_b_0", shape=(64, ), dtype="float32")    call_72 = pre_input if pre_input is not None else relay.var("call_72", shape=(1, 64, 13, 13), dtype="float32")    call_output0 = relay.nn.bias_add(call_72, fire8_squeeze1x1_b_0)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_25(pre_input=None):    import tvm    from tvm import relay    fire8_expand1x1_w_0 = relay.var("fire8/expand1x1_w_0", shape=(256, 64, 1, 1), dtype="float32")    fire8_expand3x3_b_0 = relay.var("fire8/expand3x3_b_0", shape=(256, ), dtype="float32")    fire8_expand3x3_w_0 = relay.var("fire8/expand3x3_w_0", shape=(256, 64, 3, 3), dtype="float32")    call_73 = pre_input if pre_input is not None else relay.var("call_73", shape=(1, 64, 13, 13), dtype="float32")    fire8_expand1x1_b_0 = relay.var("fire8/expand1x1_b_0", shape=(256, ), dtype="float32")    call_74 = relay.nn.relu(call_73)
    call_75 = relay.nn.conv2d(call_74, fire8_expand1x1_w_0, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_76 = relay.nn.bias_add(call_75, fire8_expand1x1_b_0)
    call_77 = relay.nn.conv2d(call_74, fire8_expand3x3_w_0, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_78 = relay.nn.bias_add(call_77, fire8_expand3x3_b_0)
    call_79 = relay.nn.relu(call_76)
    call_80 = relay.nn.relu(call_78)
    call_output0 = relay.concatenate(relay.Tuple([call_79, call_80]), axis=1)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_26(pre_input=None):    import tvm    from tvm import relay    call_82 = pre_input if pre_input is not None else relay.var("call_82", shape=(1, 512, 13, 13), dtype="float32")    fire9_squeeze1x1_w_0 = relay.var("fire9/squeeze1x1_w_0", shape=(64, 512, 1, 1), dtype="float32")    call_output0 = relay.nn.conv2d(call_82, fire9_squeeze1x1_w_0, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_27(pre_input=None):    import tvm    from tvm import relay    call_83 = pre_input if pre_input is not None else relay.var("call_83", shape=(1, 64, 13, 13), dtype="float32")    fire9_squeeze1x1_b_0 = relay.var("fire9/squeeze1x1_b_0", shape=(64, ), dtype="float32")    call_output0 = relay.nn.bias_add(call_83, fire9_squeeze1x1_b_0)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_28(pre_input=None):    import tvm    from tvm import relay    fire9_expand3x3_b_0 = relay.var("fire9/expand3x3_b_0", shape=(256, ), dtype="float32")    fire9_expand3x3_w_0 = relay.var("fire9/expand3x3_w_0", shape=(256, 64, 3, 3), dtype="float32")    call_84 = pre_input if pre_input is not None else relay.var("call_84", shape=(1, 64, 13, 13), dtype="float32")    fire9_expand1x1_b_0 = relay.var("fire9/expand1x1_b_0", shape=(256, ), dtype="float32")    fire9_expand1x1_w_0 = relay.var("fire9/expand1x1_w_0", shape=(256, 64, 1, 1), dtype="float32")    call_85 = relay.nn.relu(call_84)
    call_86 = relay.nn.conv2d(call_85, fire9_expand1x1_w_0, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_87 = relay.nn.bias_add(call_86, fire9_expand1x1_b_0)
    call_88 = relay.nn.conv2d(call_85, fire9_expand3x3_w_0, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_89 = relay.nn.bias_add(call_88, fire9_expand3x3_b_0)
    call_90 = relay.nn.relu(call_87)
    call_91 = relay.nn.relu(call_89)
    call_output0 = relay.concatenate(relay.Tuple([call_90, call_91]), axis=1)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_29(pre_input=None):    import tvm    from tvm import relay    call_93 = pre_input if pre_input is not None else relay.var("call_93", shape=(1, 512, 13, 13), dtype="float32")    call_output0 = relay.nn.dropout(call_93)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_30(pre_input=None):    import tvm    from tvm import relay    conv10_w_0 = relay.var("conv10_w_0", shape=(1000, 512, 1, 1), dtype="float32")    call_94 = pre_input if pre_input is not None else relay.var("call_94", shape=(1, 512, 13, 13), dtype="float32")    call_output0 = relay.nn.conv2d(call_94, conv10_w_0, padding=[0, 0, 0, 0], channels=1000, kernel_size=[1, 1])
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_31(pre_input=None):    import tvm    from tvm import relay    call_96 = pre_input if pre_input is not None else relay.var("call_96", shape=(1, 1000, 13, 13), dtype="float32")    conv10_b_0 = relay.var("conv10_b_0", shape=(1000, ), dtype="float32")    call_output0 = relay.nn.bias_add(call_96, conv10_b_0)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_32(pre_input=None):    import tvm    from tvm import relay    call_97 = pre_input if pre_input is not None else relay.var("call_97", shape=(1, 1000, 13, 13), dtype="float32")    call_output0 = relay.nn.relu(call_97)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_33(pre_input=None):    import tvm    from tvm import relay    call_98 = pre_input if pre_input is not None else relay.var("call_98", shape=(1, 1000, 13, 13), dtype="float32")    call_99 = relay.nn.global_avg_pool2d(call_98)
    call_100 = relay.max(call_99, axis=[1, 2, 3], keepdims=True)
    call_output0 = relay.subtract(call_99, call_100)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def SqueezeNetv1_34(pre_input=None):    import tvm    from tvm import relay    call_101 = pre_input if pre_input is not None else relay.var("call_101", shape=(1, 1000, 1, 1), dtype="float32")    call_102 = relay.exp(call_101)
    call_103 = relay.sum(call_102, axis=[1, 2, 3], keepdims=True)
    call_output0 = relay.divide(call_102, call_103)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]