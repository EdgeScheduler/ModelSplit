import tvm
    call_224 = relay.nn.batch_flatten(call_223)
    call_225 = relay.nn.dense(call_224, resnetv24_dense0_weight, units=1000)
    call_226 = relay.multiply(relay.const(1.0, dtype="float32"), resnetv24_dense0_bias)
    call_output0 = relay.add(call_225, call_226)
    return call_output0