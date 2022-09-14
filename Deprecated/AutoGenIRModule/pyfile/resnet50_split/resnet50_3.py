import tvmfrom tvm import relay, IRModuleimport numpy as npdef ResnetModule_3():    call_222 = relay.var("call_222", shape=(1, 2048, 1, 1), dtype="float32")    resnetv24_dense0_bias = relay.var("resnetv24_dense0_bias", shape=(1000, ), dtype="float32")    resnetv24_dense0_weight = relay.var("resnetv24_dense0_weight", shape=(1000, 2048), dtype="float32")    call_223 = relay.reshape(call_222, newshape=[0, -1])
    call_224 = relay.nn.batch_flatten(call_223)
    call_225 = relay.nn.dense(call_224, resnetv24_dense0_weight, units=1000)
    call_226 = relay.multiply(relay.const(1.0, dtype="float32"), resnetv24_dense0_bias)
    call_output0 = relay.add(call_225, call_226)
    return call_output0