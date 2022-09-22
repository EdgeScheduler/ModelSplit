# this file is create by program.def GoogleNet_0(pre_input=None):    import tvm    import numpy as np    from tvm import relay    data_0 = pre_input if pre_input is not None else relay.var("data_0", shape=(15, 3, 224, 224), dtype="float32")    call_0 = relay.take(data_0, relay.const(np.array(0, dtype="int64")), axis=1)
    call_1 = relay.expand_dims(call_0, axis=1)
    call_2 = relay.multiply(call_1, relay.const(0.458, dtype="float32"))
    call_3 = relay.take(data_0, relay.const(np.array(1, dtype="int64")), axis=1)
    call_4 = relay.expand_dims(call_3, axis=1)
    call_5 = relay.multiply(call_4, relay.const(0.448, dtype="float32"))
    call_6 = relay.take(data_0, relay.const(np.array(2, dtype="int64")), axis=1)
    call_7 = relay.expand_dims(call_6, axis=1)
    call_8 = relay.multiply(call_7, relay.const(0.45, dtype="float32"))
    call_9 = relay.add(call_2, relay.const(-0.03, dtype="float32"))
    call_10 = relay.add(call_5, relay.const(-0.088, dtype="float32"))
    call_11 = relay.add(call_8, relay.const(-0.188, dtype="float32"))
    call_output0 = relay.concatenate(relay.Tuple([call_9, call_10, call_11]), axis=1)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def GoogleNet_1(pre_input=None):    import tvm    import numpy as np    from tvm import relay    onnx_Conv_667 = relay.var("onnx_Conv_667", shape=(112, ), dtype="float32")    onnx_Conv_729 = relay.var("onnx_Conv_729", shape=(48, 832, 1, 1), dtype="float32")    onnx_Conv_684 = relay.var("onnx_Conv_684", shape=(256, 528, 1, 1), dtype="float32")    onnx_Conv_735 = relay.var("onnx_Conv_735", shape=(128, 832, 1, 1), dtype="float32")    onnx_Conv_580 = relay.var("onnx_Conv_580", shape=(96, ), dtype="float32")    onnx_Conv_624 = relay.var("onnx_Conv_624", shape=(48, 16, 3, 3), dtype="float32")    onnx_Conv_678 = relay.var("onnx_Conv_678", shape=(64, 32, 3, 3), dtype="float32")    onnx_Conv_699 = relay.var("onnx_Conv_699", shape=(128, 528, 1, 1), dtype="float32")    onnx_Conv_652 = relay.var("onnx_Conv_652", shape=(128, ), dtype="float32")    onnx_Conv_646 = relay.var("onnx_Conv_646", shape=(64, ), dtype="float32")    onnx_Conv_598 = relay.var("onnx_Conv_598", shape=(128, ), dtype="float32")    onnx_Conv_613 = relay.var("onnx_Conv_613", shape=(192, ), dtype="float32")    onnx_Conv_663 = relay.var("onnx_Conv_663", shape=(64, 512, 1, 1), dtype="float32")    onnx_Conv_709 = relay.var("onnx_Conv_709", shape=(320, ), dtype="float32")    onnx_Conv_640 = relay.var("onnx_Conv_640", shape=(24, ), dtype="float32")    onnx_Conv_708 = relay.var("onnx_Conv_708", shape=(320, 160, 3, 3), dtype="float32")    onnx_Conv_696 = relay.var("onnx_Conv_696", shape=(128, 32, 3, 3), dtype="float32")    onnx_Conv_601 = relay.var("onnx_Conv_601", shape=(192, ), dtype="float32")    onnx_Conv_609 = relay.var("onnx_Conv_609", shape=(64, 256, 1, 1), dtype="float32")    onnx_Conv_705 = relay.var("onnx_Conv_705", shape=(160, 832, 1, 1), dtype="float32")    call_13 = pre_input if pre_input is not None else relay.var("call_13", shape=(15, 3, 224, 224), dtype="float32")    onnx_Conv_585 = relay.var("onnx_Conv_585", shape=(16, 192, 1, 1), dtype="float32")    onnx_Conv_568 = relay.var("onnx_Conv_568", shape=(64, ), dtype="float32")    onnx_Conv_591 = relay.var("onnx_Conv_591", shape=(32, 192, 1, 1), dtype="float32")    onnx_Conv_736 = relay.var("onnx_Conv_736", shape=(128, ), dtype="float32")    call_output0 = relay.nn.conv2d(call_13, call_onnx_Conv_567, strides=[2, 2], padding=[3, 3, 3, 3], channels=64, kernel_size=[7, 7])
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def GoogleNet_2(pre_input=None):    import tvm    import numpy as np    from tvm import relay    onnx_Conv_667 = relay.var("onnx_Conv_667", shape=(112, ), dtype="float32")    onnx_Conv_729 = relay.var("onnx_Conv_729", shape=(48, 832, 1, 1), dtype="float32")    onnx_Conv_684 = relay.var("onnx_Conv_684", shape=(256, 528, 1, 1), dtype="float32")    onnx_Conv_735 = relay.var("onnx_Conv_735", shape=(128, 832, 1, 1), dtype="float32")    onnx_Conv_580 = relay.var("onnx_Conv_580", shape=(96, ), dtype="float32")    onnx_Conv_624 = relay.var("onnx_Conv_624", shape=(48, 16, 3, 3), dtype="float32")    onnx_Conv_678 = relay.var("onnx_Conv_678", shape=(64, 32, 3, 3), dtype="float32")    onnx_Conv_699 = relay.var("onnx_Conv_699", shape=(128, 528, 1, 1), dtype="float32")    onnx_Conv_652 = relay.var("onnx_Conv_652", shape=(128, ), dtype="float32")    onnx_Conv_646 = relay.var("onnx_Conv_646", shape=(64, ), dtype="float32")    onnx_Conv_598 = relay.var("onnx_Conv_598", shape=(128, ), dtype="float32")    onnx_Conv_613 = relay.var("onnx_Conv_613", shape=(192, ), dtype="float32")    onnx_Conv_663 = relay.var("onnx_Conv_663", shape=(64, 512, 1, 1), dtype="float32")    onnx_Conv_709 = relay.var("onnx_Conv_709", shape=(320, ), dtype="float32")    onnx_Conv_640 = relay.var("onnx_Conv_640", shape=(24, ), dtype="float32")    onnx_Conv_708 = relay.var("onnx_Conv_708", shape=(320, 160, 3, 3), dtype="float32")    onnx_Conv_696 = relay.var("onnx_Conv_696", shape=(128, 32, 3, 3), dtype="float32")    onnx_Conv_601 = relay.var("onnx_Conv_601", shape=(192, ), dtype="float32")    onnx_Conv_609 = relay.var("onnx_Conv_609", shape=(64, 256, 1, 1), dtype="float32")    onnx_Conv_705 = relay.var("onnx_Conv_705", shape=(160, 832, 1, 1), dtype="float32")    onnx_Conv_585 = relay.var("onnx_Conv_585", shape=(16, 192, 1, 1), dtype="float32")    onnx_Conv_568 = relay.var("onnx_Conv_568", shape=(64, ), dtype="float32")    onnx_Conv_591 = relay.var("onnx_Conv_591", shape=(32, 192, 1, 1), dtype="float32")    onnx_Conv_736 = relay.var("onnx_Conv_736", shape=(128, ), dtype="float32")    call_output0 = relay.nn.bias_add(call_14, onnx_Conv_568)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def GoogleNet_3(pre_input=None):    import tvm    import numpy as np    from tvm import relay    call_15 = pre_input if pre_input is not None else relay.var("call_15", shape=(15, 64, 112, 112), dtype="float32")    call_output0 = relay.nn.relu(call_15)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def GoogleNet_4(pre_input=None):    import tvm    import numpy as np    from tvm import relay    call_16 = pre_input if pre_input is not None else relay.var("call_16", shape=(15, 64, 112, 112), dtype="float32")    call_output0 = relay.nn.max_pool2d(call_16, pool_size=[3, 3], strides=[2, 2], padding=[0, 0, 0, 0], ceil_mode=True)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def GoogleNet_5(pre_input=None):    import tvm    import numpy as np    from tvm import relay    onnx_Conv_667 = relay.var("onnx_Conv_667", shape=(112, ), dtype="float32")    onnx_Conv_729 = relay.var("onnx_Conv_729", shape=(48, 832, 1, 1), dtype="float32")    onnx_Conv_684 = relay.var("onnx_Conv_684", shape=(256, 528, 1, 1), dtype="float32")    onnx_Conv_735 = relay.var("onnx_Conv_735", shape=(128, 832, 1, 1), dtype="float32")    onnx_Conv_580 = relay.var("onnx_Conv_580", shape=(96, ), dtype="float32")    onnx_Conv_624 = relay.var("onnx_Conv_624", shape=(48, 16, 3, 3), dtype="float32")    onnx_Conv_678 = relay.var("onnx_Conv_678", shape=(64, 32, 3, 3), dtype="float32")    onnx_Conv_699 = relay.var("onnx_Conv_699", shape=(128, 528, 1, 1), dtype="float32")    onnx_Conv_652 = relay.var("onnx_Conv_652", shape=(128, ), dtype="float32")    onnx_Conv_646 = relay.var("onnx_Conv_646", shape=(64, ), dtype="float32")    onnx_Conv_598 = relay.var("onnx_Conv_598", shape=(128, ), dtype="float32")    onnx_Conv_613 = relay.var("onnx_Conv_613", shape=(192, ), dtype="float32")    onnx_Conv_663 = relay.var("onnx_Conv_663", shape=(64, 512, 1, 1), dtype="float32")    onnx_Conv_709 = relay.var("onnx_Conv_709", shape=(320, ), dtype="float32")    onnx_Conv_640 = relay.var("onnx_Conv_640", shape=(24, ), dtype="float32")    onnx_Conv_708 = relay.var("onnx_Conv_708", shape=(320, 160, 3, 3), dtype="float32")    onnx_Conv_696 = relay.var("onnx_Conv_696", shape=(128, 32, 3, 3), dtype="float32")    onnx_Conv_601 = relay.var("onnx_Conv_601", shape=(192, ), dtype="float32")    onnx_Conv_609 = relay.var("onnx_Conv_609", shape=(64, 256, 1, 1), dtype="float32")    onnx_Conv_705 = relay.var("onnx_Conv_705", shape=(160, 832, 1, 1), dtype="float32")    onnx_Conv_585 = relay.var("onnx_Conv_585", shape=(16, 192, 1, 1), dtype="float32")    onnx_Conv_568 = relay.var("onnx_Conv_568", shape=(64, ), dtype="float32")    onnx_Conv_591 = relay.var("onnx_Conv_591", shape=(32, 192, 1, 1), dtype="float32")    onnx_Conv_736 = relay.var("onnx_Conv_736", shape=(128, ), dtype="float32")    call_output0 = relay.nn.conv2d(call_17, call_onnx_Conv_570, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def GoogleNet_6(pre_input=None):    import tvm    import numpy as np    from tvm import relay    onnx_Conv_667 = relay.var("onnx_Conv_667", shape=(112, ), dtype="float32")    onnx_Conv_729 = relay.var("onnx_Conv_729", shape=(48, 832, 1, 1), dtype="float32")    onnx_Conv_684 = relay.var("onnx_Conv_684", shape=(256, 528, 1, 1), dtype="float32")    onnx_Conv_735 = relay.var("onnx_Conv_735", shape=(128, 832, 1, 1), dtype="float32")    onnx_Conv_580 = relay.var("onnx_Conv_580", shape=(96, ), dtype="float32")    onnx_Conv_624 = relay.var("onnx_Conv_624", shape=(48, 16, 3, 3), dtype="float32")    onnx_Conv_678 = relay.var("onnx_Conv_678", shape=(64, 32, 3, 3), dtype="float32")    onnx_Conv_699 = relay.var("onnx_Conv_699", shape=(128, 528, 1, 1), dtype="float32")    onnx_Conv_652 = relay.var("onnx_Conv_652", shape=(128, ), dtype="float32")    onnx_Conv_646 = relay.var("onnx_Conv_646", shape=(64, ), dtype="float32")    onnx_Conv_598 = relay.var("onnx_Conv_598", shape=(128, ), dtype="float32")    onnx_Conv_613 = relay.var("onnx_Conv_613", shape=(192, ), dtype="float32")    onnx_Conv_663 = relay.var("onnx_Conv_663", shape=(64, 512, 1, 1), dtype="float32")    onnx_Conv_709 = relay.var("onnx_Conv_709", shape=(320, ), dtype="float32")    onnx_Conv_640 = relay.var("onnx_Conv_640", shape=(24, ), dtype="float32")    onnx_Conv_708 = relay.var("onnx_Conv_708", shape=(320, 160, 3, 3), dtype="float32")    onnx_Conv_696 = relay.var("onnx_Conv_696", shape=(128, 32, 3, 3), dtype="float32")    onnx_Conv_601 = relay.var("onnx_Conv_601", shape=(192, ), dtype="float32")    onnx_Conv_609 = relay.var("onnx_Conv_609", shape=(64, 256, 1, 1), dtype="float32")    onnx_Conv_705 = relay.var("onnx_Conv_705", shape=(160, 832, 1, 1), dtype="float32")    onnx_Conv_585 = relay.var("onnx_Conv_585", shape=(16, 192, 1, 1), dtype="float32")    onnx_Conv_568 = relay.var("onnx_Conv_568", shape=(64, ), dtype="float32")    onnx_Conv_591 = relay.var("onnx_Conv_591", shape=(32, 192, 1, 1), dtype="float32")    onnx_Conv_736 = relay.var("onnx_Conv_736", shape=(128, ), dtype="float32")    call_output0 = relay.nn.bias_add(call_18, call_onnx_Conv_571)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def GoogleNet_7(pre_input=None):    import tvm    import numpy as np    from tvm import relay    call_19 = pre_input if pre_input is not None else relay.var("call_19", shape=(15, 64, 56, 56), dtype="float32")    call_output0 = relay.nn.relu(call_19)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def GoogleNet_8(pre_input=None):    import tvm    import numpy as np    from tvm import relay    onnx_Conv_667 = relay.var("onnx_Conv_667", shape=(112, ), dtype="float32")    onnx_Conv_729 = relay.var("onnx_Conv_729", shape=(48, 832, 1, 1), dtype="float32")    onnx_Conv_684 = relay.var("onnx_Conv_684", shape=(256, 528, 1, 1), dtype="float32")    onnx_Conv_735 = relay.var("onnx_Conv_735", shape=(128, 832, 1, 1), dtype="float32")    onnx_Conv_580 = relay.var("onnx_Conv_580", shape=(96, ), dtype="float32")    onnx_Conv_624 = relay.var("onnx_Conv_624", shape=(48, 16, 3, 3), dtype="float32")    onnx_Conv_678 = relay.var("onnx_Conv_678", shape=(64, 32, 3, 3), dtype="float32")    onnx_Conv_699 = relay.var("onnx_Conv_699", shape=(128, 528, 1, 1), dtype="float32")    onnx_Conv_652 = relay.var("onnx_Conv_652", shape=(128, ), dtype="float32")    onnx_Conv_646 = relay.var("onnx_Conv_646", shape=(64, ), dtype="float32")    onnx_Conv_598 = relay.var("onnx_Conv_598", shape=(128, ), dtype="float32")    onnx_Conv_613 = relay.var("onnx_Conv_613", shape=(192, ), dtype="float32")    onnx_Conv_663 = relay.var("onnx_Conv_663", shape=(64, 512, 1, 1), dtype="float32")    onnx_Conv_709 = relay.var("onnx_Conv_709", shape=(320, ), dtype="float32")    onnx_Conv_640 = relay.var("onnx_Conv_640", shape=(24, ), dtype="float32")    onnx_Conv_708 = relay.var("onnx_Conv_708", shape=(320, 160, 3, 3), dtype="float32")    onnx_Conv_696 = relay.var("onnx_Conv_696", shape=(128, 32, 3, 3), dtype="float32")    onnx_Conv_601 = relay.var("onnx_Conv_601", shape=(192, ), dtype="float32")    onnx_Conv_609 = relay.var("onnx_Conv_609", shape=(64, 256, 1, 1), dtype="float32")    onnx_Conv_705 = relay.var("onnx_Conv_705", shape=(160, 832, 1, 1), dtype="float32")    onnx_Conv_585 = relay.var("onnx_Conv_585", shape=(16, 192, 1, 1), dtype="float32")    onnx_Conv_568 = relay.var("onnx_Conv_568", shape=(64, ), dtype="float32")    onnx_Conv_591 = relay.var("onnx_Conv_591", shape=(32, 192, 1, 1), dtype="float32")    onnx_Conv_736 = relay.var("onnx_Conv_736", shape=(128, ), dtype="float32")    call_output0 = relay.nn.conv2d(call_20, call_onnx_Conv_573, padding=[1, 1, 1, 1], channels=192, kernel_size=[3, 3])
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def GoogleNet_9(pre_input=None):    import tvm    import numpy as np    from tvm import relay    onnx_Conv_667 = relay.var("onnx_Conv_667", shape=(112, ), dtype="float32")    onnx_Conv_729 = relay.var("onnx_Conv_729", shape=(48, 832, 1, 1), dtype="float32")    onnx_Conv_684 = relay.var("onnx_Conv_684", shape=(256, 528, 1, 1), dtype="float32")    onnx_Conv_735 = relay.var("onnx_Conv_735", shape=(128, 832, 1, 1), dtype="float32")    onnx_Conv_580 = relay.var("onnx_Conv_580", shape=(96, ), dtype="float32")    onnx_Conv_624 = relay.var("onnx_Conv_624", shape=(48, 16, 3, 3), dtype="float32")    onnx_Conv_678 = relay.var("onnx_Conv_678", shape=(64, 32, 3, 3), dtype="float32")    onnx_Conv_699 = relay.var("onnx_Conv_699", shape=(128, 528, 1, 1), dtype="float32")    onnx_Conv_652 = relay.var("onnx_Conv_652", shape=(128, ), dtype="float32")    onnx_Conv_646 = relay.var("onnx_Conv_646", shape=(64, ), dtype="float32")    call_21 = pre_input if pre_input is not None else relay.var("call_21", shape=(15, 192, 56, 56), dtype="float32")    onnx_Conv_598 = relay.var("onnx_Conv_598", shape=(128, ), dtype="float32")    onnx_Conv_613 = relay.var("onnx_Conv_613", shape=(192, ), dtype="float32")    onnx_Conv_663 = relay.var("onnx_Conv_663", shape=(64, 512, 1, 1), dtype="float32")    onnx_Conv_709 = relay.var("onnx_Conv_709", shape=(320, ), dtype="float32")    onnx_Conv_640 = relay.var("onnx_Conv_640", shape=(24, ), dtype="float32")    onnx_Conv_708 = relay.var("onnx_Conv_708", shape=(320, 160, 3, 3), dtype="float32")    onnx_Conv_696 = relay.var("onnx_Conv_696", shape=(128, 32, 3, 3), dtype="float32")    onnx_Conv_601 = relay.var("onnx_Conv_601", shape=(192, ), dtype="float32")    onnx_Conv_609 = relay.var("onnx_Conv_609", shape=(64, 256, 1, 1), dtype="float32")    onnx_Conv_705 = relay.var("onnx_Conv_705", shape=(160, 832, 1, 1), dtype="float32")    onnx_Conv_585 = relay.var("onnx_Conv_585", shape=(16, 192, 1, 1), dtype="float32")    onnx_Conv_568 = relay.var("onnx_Conv_568", shape=(64, ), dtype="float32")    onnx_Conv_591 = relay.var("onnx_Conv_591", shape=(32, 192, 1, 1), dtype="float32")    onnx_Conv_736 = relay.var("onnx_Conv_736", shape=(128, ), dtype="float32")    call_output0 = relay.nn.bias_add(call_21, call_onnx_Conv_574)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def GoogleNet_10(pre_input=None):    import tvm    import numpy as np    from tvm import relay    call_22 = pre_input if pre_input is not None else relay.var("call_22", shape=(15, 192, 56, 56), dtype="float32")    call_output0 = relay.nn.relu(call_22)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def GoogleNet_11(pre_input=None):    import tvm    import numpy as np    from tvm import relay    onnx_Conv_667 = relay.var("onnx_Conv_667", shape=(112, ), dtype="float32")    onnx_Conv_729 = relay.var("onnx_Conv_729", shape=(48, 832, 1, 1), dtype="float32")    onnx_Conv_684 = relay.var("onnx_Conv_684", shape=(256, 528, 1, 1), dtype="float32")    onnx_Conv_735 = relay.var("onnx_Conv_735", shape=(128, 832, 1, 1), dtype="float32")    onnx_Conv_580 = relay.var("onnx_Conv_580", shape=(96, ), dtype="float32")    onnx_Conv_624 = relay.var("onnx_Conv_624", shape=(48, 16, 3, 3), dtype="float32")    onnx_Conv_678 = relay.var("onnx_Conv_678", shape=(64, 32, 3, 3), dtype="float32")    onnx_Conv_699 = relay.var("onnx_Conv_699", shape=(128, 528, 1, 1), dtype="float32")    onnx_Conv_652 = relay.var("onnx_Conv_652", shape=(128, ), dtype="float32")    onnx_Conv_646 = relay.var("onnx_Conv_646", shape=(64, ), dtype="float32")    onnx_Conv_598 = relay.var("onnx_Conv_598", shape=(128, ), dtype="float32")    call_23 = pre_input if pre_input is not None else relay.var("call_23", shape=(15, 192, 56, 56), dtype="float32")    onnx_Conv_613 = relay.var("onnx_Conv_613", shape=(192, ), dtype="float32")    onnx_Conv_663 = relay.var("onnx_Conv_663", shape=(64, 512, 1, 1), dtype="float32")    onnx_Conv_709 = relay.var("onnx_Conv_709", shape=(320, ), dtype="float32")    onnx_Conv_640 = relay.var("onnx_Conv_640", shape=(24, ), dtype="float32")    onnx_Conv_708 = relay.var("onnx_Conv_708", shape=(320, 160, 3, 3), dtype="float32")    onnx_Conv_696 = relay.var("onnx_Conv_696", shape=(128, 32, 3, 3), dtype="float32")    onnx_Conv_601 = relay.var("onnx_Conv_601", shape=(192, ), dtype="float32")    onnx_Conv_609 = relay.var("onnx_Conv_609", shape=(64, 256, 1, 1), dtype="float32")    onnx_Conv_705 = relay.var("onnx_Conv_705", shape=(160, 832, 1, 1), dtype="float32")    onnx_Conv_585 = relay.var("onnx_Conv_585", shape=(16, 192, 1, 1), dtype="float32")    onnx_Conv_568 = relay.var("onnx_Conv_568", shape=(64, ), dtype="float32")    onnx_Conv_591 = relay.var("onnx_Conv_591", shape=(32, 192, 1, 1), dtype="float32")    onnx_Conv_736 = relay.var("onnx_Conv_736", shape=(128, ), dtype="float32")    call_24 = relay.nn.max_pool2d(call_23, pool_size=[3, 3], strides=[2, 2], padding=[0, 0, 0, 0], ceil_mode=True)
    call_25 = relay.nn.conv2d(call_24, call_onnx_Conv_576, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_26 = relay.nn.bias_add(call_25, call_onnx_Conv_577)
    call_27 = relay.nn.conv2d(call_24, call_onnx_Conv_579, padding=[0, 0, 0, 0], channels=96, kernel_size=[1, 1])
    call_28 = relay.nn.bias_add(call_27, onnx_Conv_580)
    call_29 = relay.nn.relu(call_28)
    call_30 = relay.nn.conv2d(call_29, call_onnx_Conv_582, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_31 = relay.nn.bias_add(call_30, call_onnx_Conv_583)
    call_32 = relay.nn.conv2d(call_24, onnx_Conv_585, padding=[0, 0, 0, 0], channels=16, kernel_size=[1, 1])
    call_33 = relay.nn.bias_add(call_32, call_onnx_Conv_586)
    call_34 = relay.nn.relu(call_33)
    call_35 = relay.nn.conv2d(call_34, call_onnx_Conv_588, padding=[1, 1, 1, 1], channels=32, kernel_size=[3, 3])
    call_36 = relay.nn.bias_add(call_35, call_onnx_Conv_589)
    call_37 = relay.nn.max_pool2d(call_24, pool_size=[3, 3], padding=[1, 1, 1, 1], ceil_mode=True)
    call_38 = relay.nn.conv2d(call_37, onnx_Conv_591, padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1])
    call_39 = relay.nn.bias_add(call_38, call_onnx_Conv_592)
    call_40 = relay.nn.relu(call_26)
    call_41 = relay.nn.relu(call_31)
    call_42 = relay.nn.relu(call_36)
    call_43 = relay.nn.relu(call_39)
    call_45 = relay.concatenate(relay.Tuple([call_40, call_41, call_42, call_43]), axis=1)
    call_46 = relay.nn.conv2d(call_45, call_onnx_Conv_594, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_47 = relay.nn.bias_add(call_46, call_onnx_Conv_595)
    call_48 = relay.nn.conv2d(call_45, call_onnx_Conv_597, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_49 = relay.nn.bias_add(call_48, onnx_Conv_598)
    call_50 = relay.nn.relu(call_49)
    call_51 = relay.nn.conv2d(call_50, call_onnx_Conv_600, padding=[1, 1, 1, 1], channels=192, kernel_size=[3, 3])
    call_52 = relay.nn.bias_add(call_51, onnx_Conv_601)
    call_53 = relay.nn.conv2d(call_45, call_onnx_Conv_603, padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1])
    call_54 = relay.nn.bias_add(call_53, call_onnx_Conv_604)
    call_55 = relay.nn.relu(call_54)
    call_56 = relay.nn.conv2d(call_55, call_onnx_Conv_606, padding=[1, 1, 1, 1], channels=96, kernel_size=[3, 3])
    call_57 = relay.nn.bias_add(call_56, call_onnx_Conv_607)
    call_58 = relay.nn.max_pool2d(call_45, pool_size=[3, 3], padding=[1, 1, 1, 1], ceil_mode=True)
    call_59 = relay.nn.conv2d(call_58, onnx_Conv_609, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_60 = relay.nn.bias_add(call_59, call_onnx_Conv_610)
    call_61 = relay.nn.relu(call_47)
    call_62 = relay.nn.relu(call_52)
    call_63 = relay.nn.relu(call_57)
    call_64 = relay.nn.relu(call_60)
    call_output0 = relay.concatenate(relay.Tuple([call_61, call_62, call_63, call_64]), axis=1)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def GoogleNet_12(pre_input=None):    import tvm    import numpy as np    from tvm import relay    onnx_Conv_667 = relay.var("onnx_Conv_667", shape=(112, ), dtype="float32")    onnx_Conv_729 = relay.var("onnx_Conv_729", shape=(48, 832, 1, 1), dtype="float32")    onnx_Conv_684 = relay.var("onnx_Conv_684", shape=(256, 528, 1, 1), dtype="float32")    onnx_Conv_735 = relay.var("onnx_Conv_735", shape=(128, 832, 1, 1), dtype="float32")    onnx_Conv_580 = relay.var("onnx_Conv_580", shape=(96, ), dtype="float32")    onnx_Conv_624 = relay.var("onnx_Conv_624", shape=(48, 16, 3, 3), dtype="float32")    onnx_Conv_678 = relay.var("onnx_Conv_678", shape=(64, 32, 3, 3), dtype="float32")    onnx_Conv_699 = relay.var("onnx_Conv_699", shape=(128, 528, 1, 1), dtype="float32")    onnx_Conv_652 = relay.var("onnx_Conv_652", shape=(128, ), dtype="float32")    onnx_Conv_646 = relay.var("onnx_Conv_646", shape=(64, ), dtype="float32")    onnx_Conv_598 = relay.var("onnx_Conv_598", shape=(128, ), dtype="float32")    onnx_Conv_613 = relay.var("onnx_Conv_613", shape=(192, ), dtype="float32")    onnx_Conv_663 = relay.var("onnx_Conv_663", shape=(64, 512, 1, 1), dtype="float32")    onnx_Conv_709 = relay.var("onnx_Conv_709", shape=(320, ), dtype="float32")    onnx_Conv_640 = relay.var("onnx_Conv_640", shape=(24, ), dtype="float32")    onnx_Conv_708 = relay.var("onnx_Conv_708", shape=(320, 160, 3, 3), dtype="float32")    onnx_Conv_696 = relay.var("onnx_Conv_696", shape=(128, 32, 3, 3), dtype="float32")    onnx_Conv_601 = relay.var("onnx_Conv_601", shape=(192, ), dtype="float32")    onnx_Conv_609 = relay.var("onnx_Conv_609", shape=(64, 256, 1, 1), dtype="float32")    onnx_Conv_705 = relay.var("onnx_Conv_705", shape=(160, 832, 1, 1), dtype="float32")    onnx_Conv_585 = relay.var("onnx_Conv_585", shape=(16, 192, 1, 1), dtype="float32")    onnx_Conv_568 = relay.var("onnx_Conv_568", shape=(64, ), dtype="float32")    onnx_Conv_591 = relay.var("onnx_Conv_591", shape=(32, 192, 1, 1), dtype="float32")    onnx_Conv_736 = relay.var("onnx_Conv_736", shape=(128, ), dtype="float32")    call_67 = relay.nn.max_pool2d(call_66, pool_size=[3, 3], strides=[2, 2], padding=[0, 0, 0, 0], ceil_mode=True)
    call_68 = relay.nn.conv2d(call_67, call_onnx_Conv_612, padding=[0, 0, 0, 0], channels=192, kernel_size=[1, 1])
    call_69 = relay.nn.bias_add(call_68, onnx_Conv_613)
    call_70 = relay.nn.conv2d(call_67, call_onnx_Conv_615, padding=[0, 0, 0, 0], channels=96, kernel_size=[1, 1])
    call_71 = relay.nn.bias_add(call_70, call_onnx_Conv_616)
    call_72 = relay.nn.relu(call_71)
    call_73 = relay.nn.conv2d(call_72, call_onnx_Conv_618, padding=[1, 1, 1, 1], channels=208, kernel_size=[3, 3])
    call_74 = relay.nn.bias_add(call_73, call_onnx_Conv_619)
    call_75 = relay.nn.conv2d(call_67, call_onnx_Conv_621, padding=[0, 0, 0, 0], channels=16, kernel_size=[1, 1])
    call_76 = relay.nn.bias_add(call_75, call_onnx_Conv_622)
    call_77 = relay.nn.relu(call_76)
    call_78 = relay.nn.conv2d(call_77, onnx_Conv_624, padding=[1, 1, 1, 1], channels=48, kernel_size=[3, 3])
    call_79 = relay.nn.bias_add(call_78, call_onnx_Conv_625)
    call_80 = relay.nn.max_pool2d(call_67, pool_size=[3, 3], padding=[1, 1, 1, 1], ceil_mode=True)
    call_81 = relay.nn.conv2d(call_80, call_onnx_Conv_627, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_82 = relay.nn.bias_add(call_81, call_onnx_Conv_628)
    call_83 = relay.nn.relu(call_69)
    call_84 = relay.nn.relu(call_74)
    call_85 = relay.nn.relu(call_79)
    call_86 = relay.nn.relu(call_82)
    call_88 = relay.concatenate(relay.Tuple([call_83, call_84, call_85, call_86]), axis=1)
    call_89 = relay.nn.conv2d(call_88, call_onnx_Conv_630, padding=[0, 0, 0, 0], channels=160, kernel_size=[1, 1])
    call_90 = relay.nn.bias_add(call_89, call_onnx_Conv_631)
    call_91 = relay.nn.conv2d(call_88, call_onnx_Conv_633, padding=[0, 0, 0, 0], channels=112, kernel_size=[1, 1])
    call_92 = relay.nn.bias_add(call_91, call_onnx_Conv_634)
    call_93 = relay.nn.relu(call_92)
    call_94 = relay.nn.conv2d(call_93, call_onnx_Conv_636, padding=[1, 1, 1, 1], channels=224, kernel_size=[3, 3])
    call_95 = relay.nn.bias_add(call_94, call_onnx_Conv_637)
    call_96 = relay.nn.conv2d(call_88, call_onnx_Conv_639, padding=[0, 0, 0, 0], channels=24, kernel_size=[1, 1])
    call_97 = relay.nn.bias_add(call_96, onnx_Conv_640)
    call_98 = relay.nn.relu(call_97)
    call_99 = relay.nn.conv2d(call_98, call_onnx_Conv_642, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_100 = relay.nn.bias_add(call_99, call_onnx_Conv_643)
    call_101 = relay.nn.max_pool2d(call_88, pool_size=[3, 3], padding=[1, 1, 1, 1], ceil_mode=True)
    call_102 = relay.nn.conv2d(call_101, call_onnx_Conv_645, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_103 = relay.nn.bias_add(call_102, onnx_Conv_646)
    call_104 = relay.nn.relu(call_90)
    call_105 = relay.nn.relu(call_95)
    call_106 = relay.nn.relu(call_100)
    call_107 = relay.nn.relu(call_103)
    call_109 = relay.concatenate(relay.Tuple([call_104, call_105, call_106, call_107]), axis=1)
    call_110 = relay.nn.conv2d(call_109, call_onnx_Conv_648, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_111 = relay.nn.bias_add(call_110, call_onnx_Conv_649)
    call_112 = relay.nn.conv2d(call_109, call_onnx_Conv_651, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_113 = relay.nn.bias_add(call_112, onnx_Conv_652)
    call_114 = relay.nn.relu(call_113)
    call_115 = relay.nn.conv2d(call_114, call_onnx_Conv_654, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_116 = relay.nn.bias_add(call_115, call_onnx_Conv_655)
    call_117 = relay.nn.conv2d(call_109, call_onnx_Conv_657, padding=[0, 0, 0, 0], channels=24, kernel_size=[1, 1])
    call_118 = relay.nn.bias_add(call_117, call_onnx_Conv_658)
    call_119 = relay.nn.relu(call_118)
    call_120 = relay.nn.conv2d(call_119, call_onnx_Conv_660, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_121 = relay.nn.bias_add(call_120, call_onnx_Conv_661)
    call_122 = relay.nn.max_pool2d(call_109, pool_size=[3, 3], padding=[1, 1, 1, 1], ceil_mode=True)
    call_123 = relay.nn.conv2d(call_122, onnx_Conv_663, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_124 = relay.nn.bias_add(call_123, call_onnx_Conv_664)
    call_125 = relay.nn.relu(call_111)
    call_126 = relay.nn.relu(call_116)
    call_127 = relay.nn.relu(call_121)
    call_128 = relay.nn.relu(call_124)
    call_130 = relay.concatenate(relay.Tuple([call_125, call_126, call_127, call_128]), axis=1)
    call_131 = relay.nn.conv2d(call_130, call_onnx_Conv_666, padding=[0, 0, 0, 0], channels=112, kernel_size=[1, 1])
    call_132 = relay.nn.bias_add(call_131, onnx_Conv_667)
    call_133 = relay.nn.conv2d(call_130, call_onnx_Conv_669, padding=[0, 0, 0, 0], channels=144, kernel_size=[1, 1])
    call_134 = relay.nn.bias_add(call_133, call_onnx_Conv_670)
    call_135 = relay.nn.relu(call_134)
    call_136 = relay.nn.conv2d(call_135, call_onnx_Conv_672, padding=[1, 1, 1, 1], channels=288, kernel_size=[3, 3])
    call_137 = relay.nn.bias_add(call_136, call_onnx_Conv_673)
    call_138 = relay.nn.conv2d(call_130, call_onnx_Conv_675, padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1])
    call_139 = relay.nn.bias_add(call_138, call_onnx_Conv_676)
    call_140 = relay.nn.relu(call_139)
    call_141 = relay.nn.conv2d(call_140, onnx_Conv_678, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_142 = relay.nn.bias_add(call_141, call_onnx_Conv_679)
    call_143 = relay.nn.max_pool2d(call_130, pool_size=[3, 3], padding=[1, 1, 1, 1], ceil_mode=True)
    call_144 = relay.nn.conv2d(call_143, call_onnx_Conv_681, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_145 = relay.nn.bias_add(call_144, call_onnx_Conv_682)
    call_146 = relay.nn.relu(call_132)
    call_147 = relay.nn.relu(call_137)
    call_148 = relay.nn.relu(call_142)
    call_149 = relay.nn.relu(call_145)
    call_151 = relay.concatenate(relay.Tuple([call_146, call_147, call_148, call_149]), axis=1)
    call_152 = relay.nn.conv2d(call_151, onnx_Conv_684, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_153 = relay.nn.bias_add(call_152, call_onnx_Conv_685)
    call_154 = relay.nn.conv2d(call_151, call_onnx_Conv_687, padding=[0, 0, 0, 0], channels=160, kernel_size=[1, 1])
    call_155 = relay.nn.bias_add(call_154, call_onnx_Conv_688)
    call_156 = relay.nn.relu(call_155)
    call_157 = relay.nn.conv2d(call_156, call_onnx_Conv_690, padding=[1, 1, 1, 1], channels=320, kernel_size=[3, 3])
    call_158 = relay.nn.bias_add(call_157, call_onnx_Conv_691)
    call_159 = relay.nn.conv2d(call_151, call_onnx_Conv_693, padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1])
    call_160 = relay.nn.bias_add(call_159, call_onnx_Conv_694)
    call_161 = relay.nn.relu(call_160)
    call_162 = relay.nn.conv2d(call_161, onnx_Conv_696, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_163 = relay.nn.bias_add(call_162, call_onnx_Conv_697)
    call_164 = relay.nn.max_pool2d(call_151, pool_size=[3, 3], padding=[1, 1, 1, 1], ceil_mode=True)
    call_165 = relay.nn.conv2d(call_164, onnx_Conv_699, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_166 = relay.nn.bias_add(call_165, call_onnx_Conv_700)
    call_167 = relay.nn.relu(call_153)
    call_168 = relay.nn.relu(call_158)
    call_169 = relay.nn.relu(call_163)
    call_170 = relay.nn.relu(call_166)
    call_output0 = relay.concatenate(relay.Tuple([call_167, call_168, call_169, call_170]), axis=1)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def GoogleNet_13(pre_input=None):    import tvm    import numpy as np    from tvm import relay    onnx_Conv_667 = relay.var("onnx_Conv_667", shape=(112, ), dtype="float32")    onnx_Conv_729 = relay.var("onnx_Conv_729", shape=(48, 832, 1, 1), dtype="float32")    onnx_Conv_684 = relay.var("onnx_Conv_684", shape=(256, 528, 1, 1), dtype="float32")    onnx_Conv_735 = relay.var("onnx_Conv_735", shape=(128, 832, 1, 1), dtype="float32")    onnx_Conv_580 = relay.var("onnx_Conv_580", shape=(96, ), dtype="float32")    onnx_Conv_624 = relay.var("onnx_Conv_624", shape=(48, 16, 3, 3), dtype="float32")    onnx_Conv_678 = relay.var("onnx_Conv_678", shape=(64, 32, 3, 3), dtype="float32")    onnx_Conv_699 = relay.var("onnx_Conv_699", shape=(128, 528, 1, 1), dtype="float32")    onnx_Conv_652 = relay.var("onnx_Conv_652", shape=(128, ), dtype="float32")    onnx_Conv_646 = relay.var("onnx_Conv_646", shape=(64, ), dtype="float32")    onnx_Conv_598 = relay.var("onnx_Conv_598", shape=(128, ), dtype="float32")    onnx_Conv_613 = relay.var("onnx_Conv_613", shape=(192, ), dtype="float32")    onnx_Conv_663 = relay.var("onnx_Conv_663", shape=(64, 512, 1, 1), dtype="float32")    onnx_Conv_709 = relay.var("onnx_Conv_709", shape=(320, ), dtype="float32")    onnx_Conv_640 = relay.var("onnx_Conv_640", shape=(24, ), dtype="float32")    onnx_Conv_708 = relay.var("onnx_Conv_708", shape=(320, 160, 3, 3), dtype="float32")    onnx_Conv_696 = relay.var("onnx_Conv_696", shape=(128, 32, 3, 3), dtype="float32")    call_172 = pre_input if pre_input is not None else relay.var("call_172", shape=(15, 832, 14, 14), dtype="float32")    onnx_Conv_601 = relay.var("onnx_Conv_601", shape=(192, ), dtype="float32")    onnx_Conv_609 = relay.var("onnx_Conv_609", shape=(64, 256, 1, 1), dtype="float32")    onnx_Conv_705 = relay.var("onnx_Conv_705", shape=(160, 832, 1, 1), dtype="float32")    onnx_Conv_585 = relay.var("onnx_Conv_585", shape=(16, 192, 1, 1), dtype="float32")    onnx_Conv_568 = relay.var("onnx_Conv_568", shape=(64, ), dtype="float32")    onnx_Conv_591 = relay.var("onnx_Conv_591", shape=(32, 192, 1, 1), dtype="float32")    onnx_Conv_736 = relay.var("onnx_Conv_736", shape=(128, ), dtype="float32")    call_173 = relay.nn.max_pool2d(call_172, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0], ceil_mode=True)
    call_174 = relay.nn.conv2d(call_173, call_onnx_Conv_702, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_175 = relay.nn.bias_add(call_174, call_onnx_Conv_703)
    call_176 = relay.nn.conv2d(call_173, onnx_Conv_705, padding=[0, 0, 0, 0], channels=160, kernel_size=[1, 1])
    call_177 = relay.nn.bias_add(call_176, call_onnx_Conv_706)
    call_178 = relay.nn.relu(call_177)
    call_179 = relay.nn.conv2d(call_178, onnx_Conv_708, padding=[1, 1, 1, 1], channels=320, kernel_size=[3, 3])
    call_180 = relay.nn.bias_add(call_179, onnx_Conv_709)
    call_181 = relay.nn.conv2d(call_173, call_onnx_Conv_711, padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1])
    call_182 = relay.nn.bias_add(call_181, call_onnx_Conv_712)
    call_183 = relay.nn.relu(call_182)
    call_184 = relay.nn.conv2d(call_183, call_onnx_Conv_714, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_185 = relay.nn.bias_add(call_184, call_onnx_Conv_715)
    call_186 = relay.nn.max_pool2d(call_173, pool_size=[3, 3], padding=[1, 1, 1, 1], ceil_mode=True)
    call_187 = relay.nn.conv2d(call_186, call_onnx_Conv_717, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_188 = relay.nn.bias_add(call_187, call_onnx_Conv_718)
    call_189 = relay.nn.relu(call_175)
    call_190 = relay.nn.relu(call_180)
    call_191 = relay.nn.relu(call_185)
    call_192 = relay.nn.relu(call_188)
    call_194 = relay.concatenate(relay.Tuple([call_189, call_190, call_191, call_192]), axis=1)
    call_195 = relay.nn.conv2d(call_194, call_onnx_Conv_720, padding=[0, 0, 0, 0], channels=384, kernel_size=[1, 1])
    call_196 = relay.nn.bias_add(call_195, call_onnx_Conv_721)
    call_197 = relay.nn.conv2d(call_194, call_onnx_Conv_723, padding=[0, 0, 0, 0], channels=192, kernel_size=[1, 1])
    call_198 = relay.nn.bias_add(call_197, call_onnx_Conv_724)
    call_199 = relay.nn.relu(call_198)
    call_200 = relay.nn.conv2d(call_199, call_onnx_Conv_726, padding=[1, 1, 1, 1], channels=384, kernel_size=[3, 3])
    call_201 = relay.nn.bias_add(call_200, call_onnx_Conv_727)
    call_202 = relay.nn.conv2d(call_194, onnx_Conv_729, padding=[0, 0, 0, 0], channels=48, kernel_size=[1, 1])
    call_203 = relay.nn.bias_add(call_202, call_onnx_Conv_730)
    call_204 = relay.nn.relu(call_203)
    call_205 = relay.nn.conv2d(call_204, call_onnx_Conv_732, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_206 = relay.nn.bias_add(call_205, call_onnx_Conv_733)
    call_207 = relay.nn.max_pool2d(call_194, pool_size=[3, 3], padding=[1, 1, 1, 1], ceil_mode=True)
    call_208 = relay.nn.conv2d(call_207, onnx_Conv_735, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_209 = relay.nn.bias_add(call_208, onnx_Conv_736)
    call_210 = relay.nn.relu(call_196)
    call_211 = relay.nn.relu(call_201)
    call_212 = relay.nn.relu(call_206)
    call_213 = relay.nn.relu(call_209)
    call_output0 = relay.concatenate(relay.Tuple([call_210, call_211, call_212, call_213]), axis=1)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def GoogleNet_14(pre_input=None):    import tvm    import numpy as np    from tvm import relay    call_215 = pre_input if pre_input is not None else relay.var("call_215", shape=(15, 1024, 7, 7), dtype="float32")    call_output0 = relay.nn.global_avg_pool2d(call_215)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def GoogleNet_15(pre_input=None):    import tvm    import numpy as np    from tvm import relay    call_216 = pre_input if pre_input is not None else relay.var("call_216", shape=(15, 1024, 1, 1), dtype="float32")    call_output0 = relay.nn.batch_flatten(call_216)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def GoogleNet_16(pre_input=None):    import tvm    import numpy as np    from tvm import relay    call_217 = pre_input if pre_input is not None else relay.var("call_217", shape=(15, 1024), dtype="float32")    call_output0 = relay.nn.batch_flatten(call_217)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]def GoogleNet_17(pre_input=None):    import tvm    import numpy as np    from tvm import relay    call_218 = pre_input if pre_input is not None else relay.var("call_218", shape=(15, 1024), dtype="float32")    fc_weight = relay.var("fc.weight", shape=(1000, 1024), dtype="float32")    fc_bias = relay.var("fc.bias", shape=(1000, ), dtype="float32")    call_219 = relay.nn.dense(call_218, fc_weight, units=1000)
    call_220 = relay.multiply(relay.const(1, dtype="float32"), fc_bias)
    call_output0 = relay.add(call_219, call_220)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]