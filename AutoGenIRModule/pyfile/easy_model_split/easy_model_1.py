import tvmfrom tvm import relay, IRModuleimport numpy as npdef EasyModule_1():    call_9 = relay.var("call_9", shape=(4, 1, 6, 6), dtype="float32")    add2 = relay.var("add2", shape=(4, 1, 6, 6), dtype="float32")    bias3 = relay.var("bias3", shape=(1, ), dtype="float32")    weight3 = relay.var("weight3", shape=(1, 144), dtype="float32")    call_10 = relay.add(call_9, add2)
    call_11 = relay.reshape(call_10, newshape=[1, 144])
    call_12 = relay.nn.batch_flatten(call_11)
    call_13 = relay.nn.dense(call_12, weight3, units=1, )
    call_output0 = relay.add(call_13, bias3)
    return call_output0