import tvm
    call_207 = relay.reshape(call_205_0, newshape=[1, 1024])
    call_208 = relay.nn.batch_flatten(call_207)
    call_209 = relay.nn.dense(call_208, loss3_classifier_w_0, units=1000)
    call_210 = relay.multiply(relay.const(1.0, dtype="float32"), loss3_classifier_b_0)
    call_211 = relay.add(call_209, call_210)
    call_212 = relay.max(call_211, axis=[1], keepdims=True)
    call_213 = relay.subtract(call_211, call_212)
    call_214 = relay.exp(call_213)
    call_215 = relay.sum(call_214, axis=[1], keepdims=True)
    call_output0 = relay.divide(call_214, call_215)
    return call_output0