import tvm
    call_12 = relay.nn.batch_flatten(call_11)
    call_13 = relay.nn.dense(call_12, weight3, units=1, )
    call_output0 = relay.add(call_13, bias3)
    return call_output0