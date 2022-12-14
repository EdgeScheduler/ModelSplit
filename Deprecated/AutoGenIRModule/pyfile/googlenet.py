from tvm import relay, IRModule
import numpy as np


def GoogleNetModule():
    data_0 = relay.var("data_0", shape=(1, 3, 224, 224), dtype="float32")
    conv1_7x7_s2_w_0 = relay.var(
        "conv1/7x7_s2_w_0", shape=(64, 3, 7, 7), dtype="float32")
    conv1_7x7_s2_b_0 = relay.var(
        "conv1/7x7_s2_b_0", shape=(64, ), dtype="float32")
    conv2_3x3_reduce_w_0 = relay.var(
        "conv2/3x3_reduce_w_0", shape=(64, 64, 1, 1), dtype="float32")
    conv2_3x3_reduce_b_0 = relay.var(
        "conv2/3x3_reduce_b_0", shape=(64, ), dtype="float32")
    conv2_3x3_w_0 = relay.var(
        "conv2/3x3_w_0", shape=(192, 64, 3, 3), dtype="float32")
    conv2_3x3_b_0 = relay.var("conv2/3x3_b_0", shape=(192, ), dtype="float32")
    inception_3a_1x1_w_0 = relay.var(
        "inception_3a/1x1_w_0", shape=(64, 192, 1, 1), dtype="float32")
    inception_3a_1x1_b_0 = relay.var(
        "inception_3a/1x1_b_0", shape=(64, ), dtype="float32")
    inception_3a_3x3_reduce_w_0 = relay.var(
        "inception_3a/3x3_reduce_w_0", shape=(96, 192, 1, 1), dtype="float32")
    inception_3a_3x3_reduce_b_0 = relay.var(
        "inception_3a/3x3_reduce_b_0", shape=(96, ), dtype="float32")
    inception_3a_3x3_w_0 = relay.var(
        "inception_3a/3x3_w_0", shape=(128, 96, 3, 3), dtype="float32")
    inception_3a_3x3_b_0 = relay.var(
        "inception_3a/3x3_b_0", shape=(128, ), dtype="float32")
    inception_3a_5x5_reduce_w_0 = relay.var(
        "inception_3a/5x5_reduce_w_0", shape=(16, 192, 1, 1), dtype="float32")
    inception_3a_5x5_reduce_b_0 = relay.var(
        "inception_3a/5x5_reduce_b_0", shape=(16, ), dtype="float32")
    inception_3a_5x5_w_0 = relay.var(
        "inception_3a/5x5_w_0", shape=(32, 16, 5, 5), dtype="float32")
    inception_3a_5x5_b_0 = relay.var(
        "inception_3a/5x5_b_0", shape=(32, ), dtype="float32")
    inception_3a_pool_proj_w_0 = relay.var(
        "inception_3a/pool_proj_w_0", shape=(32, 192, 1, 1), dtype="float32")
    inception_3a_pool_proj_b_0 = relay.var(
        "inception_3a/pool_proj_b_0", shape=(32, ), dtype="float32")
    inception_3b_1x1_w_0 = relay.var(
        "inception_3b/1x1_w_0", shape=(128, 256, 1, 1), dtype="float32")
    inception_3b_1x1_b_0 = relay.var(
        "inception_3b/1x1_b_0", shape=(128, ), dtype="float32")
    inception_3b_3x3_reduce_w_0 = relay.var(
        "inception_3b/3x3_reduce_w_0", shape=(128, 256, 1, 1), dtype="float32")
    inception_3b_3x3_reduce_b_0 = relay.var(
        "inception_3b/3x3_reduce_b_0", shape=(128, ), dtype="float32")
    inception_3b_3x3_w_0 = relay.var(
        "inception_3b/3x3_w_0", shape=(192, 128, 3, 3), dtype="float32")
    inception_3b_3x3_b_0 = relay.var(
        "inception_3b/3x3_b_0", shape=(192, ), dtype="float32")
    inception_3b_5x5_reduce_w_0 = relay.var(
        "inception_3b/5x5_reduce_w_0", shape=(32, 256, 1, 1), dtype="float32")
    inception_3b_5x5_reduce_b_0 = relay.var(
        "inception_3b/5x5_reduce_b_0", shape=(32, ), dtype="float32")
    inception_3b_5x5_w_0 = relay.var(
        "inception_3b/5x5_w_0", shape=(96, 32, 5, 5), dtype="float32")
    inception_3b_5x5_b_0 = relay.var(
        "inception_3b/5x5_b_0", shape=(96, ), dtype="float32")
    inception_3b_pool_proj_w_0 = relay.var(
        "inception_3b/pool_proj_w_0", shape=(64, 256, 1, 1), dtype="float32")
    inception_3b_pool_proj_b_0 = relay.var(
        "inception_3b/pool_proj_b_0", shape=(64, ), dtype="float32")
    inception_4a_1x1_w_0 = relay.var(
        "inception_4a/1x1_w_0", shape=(192, 480, 1, 1), dtype="float32")
    inception_4a_1x1_b_0 = relay.var(
        "inception_4a/1x1_b_0", shape=(192, ), dtype="float32")
    inception_4a_3x3_reduce_w_0 = relay.var(
        "inception_4a/3x3_reduce_w_0", shape=(96, 480, 1, 1), dtype="float32")
    inception_4a_3x3_reduce_b_0 = relay.var(
        "inception_4a/3x3_reduce_b_0", shape=(96, ), dtype="float32")
    inception_4a_3x3_w_0 = relay.var(
        "inception_4a/3x3_w_0", shape=(208, 96, 3, 3), dtype="float32")
    inception_4a_3x3_b_0 = relay.var(
        "inception_4a/3x3_b_0", shape=(208, ), dtype="float32")
    inception_4a_5x5_reduce_w_0 = relay.var(
        "inception_4a/5x5_reduce_w_0", shape=(16, 480, 1, 1), dtype="float32")
    inception_4a_5x5_reduce_b_0 = relay.var(
        "inception_4a/5x5_reduce_b_0", shape=(16, ), dtype="float32")
    inception_4a_5x5_w_0 = relay.var(
        "inception_4a/5x5_w_0", shape=(48, 16, 5, 5), dtype="float32")
    inception_4a_5x5_b_0 = relay.var(
        "inception_4a/5x5_b_0", shape=(48, ), dtype="float32")
    inception_4a_pool_proj_w_0 = relay.var(
        "inception_4a/pool_proj_w_0", shape=(64, 480, 1, 1), dtype="float32")
    inception_4a_pool_proj_b_0 = relay.var(
        "inception_4a/pool_proj_b_0", shape=(64, ), dtype="float32")
    inception_4b_1x1_w_0 = relay.var(
        "inception_4b/1x1_w_0", shape=(160, 512, 1, 1), dtype="float32")
    inception_4b_1x1_b_0 = relay.var(
        "inception_4b/1x1_b_0", shape=(160, ), dtype="float32")
    inception_4b_3x3_reduce_w_0 = relay.var(
        "inception_4b/3x3_reduce_w_0", shape=(112, 512, 1, 1), dtype="float32")
    inception_4b_3x3_reduce_b_0 = relay.var(
        "inception_4b/3x3_reduce_b_0", shape=(112, ), dtype="float32")
    inception_4b_3x3_w_0 = relay.var(
        "inception_4b/3x3_w_0", shape=(224, 112, 3, 3), dtype="float32")
    inception_4b_3x3_b_0 = relay.var(
        "inception_4b/3x3_b_0", shape=(224, ), dtype="float32")
    inception_4b_5x5_reduce_w_0 = relay.var(
        "inception_4b/5x5_reduce_w_0", shape=(24, 512, 1, 1), dtype="float32")
    inception_4b_5x5_reduce_b_0 = relay.var(
        "inception_4b/5x5_reduce_b_0", shape=(24, ), dtype="float32")
    inception_4b_5x5_w_0 = relay.var(
        "inception_4b/5x5_w_0", shape=(64, 24, 5, 5), dtype="float32")
    inception_4b_5x5_b_0 = relay.var(
        "inception_4b/5x5_b_0", shape=(64, ), dtype="float32")
    inception_4b_pool_proj_w_0 = relay.var(
        "inception_4b/pool_proj_w_0", shape=(64, 512, 1, 1), dtype="float32")
    inception_4b_pool_proj_b_0 = relay.var(
        "inception_4b/pool_proj_b_0", shape=(64, ), dtype="float32")
    inception_4c_1x1_w_0 = relay.var(
        "inception_4c/1x1_w_0", shape=(128, 512, 1, 1), dtype="float32")
    inception_4c_1x1_b_0 = relay.var(
        "inception_4c/1x1_b_0", shape=(128, ), dtype="float32")
    inception_4c_3x3_reduce_w_0 = relay.var(
        "inception_4c/3x3_reduce_w_0", shape=(128, 512, 1, 1), dtype="float32")
    inception_4c_3x3_reduce_b_0 = relay.var(
        "inception_4c/3x3_reduce_b_0", shape=(128, ), dtype="float32")
    inception_4c_3x3_w_0 = relay.var(
        "inception_4c/3x3_w_0", shape=(256, 128, 3, 3), dtype="float32")
    inception_4c_3x3_b_0 = relay.var(
        "inception_4c/3x3_b_0", shape=(256, ), dtype="float32")
    inception_4c_5x5_reduce_w_0 = relay.var(
        "inception_4c/5x5_reduce_w_0", shape=(24, 512, 1, 1), dtype="float32")
    inception_4c_5x5_reduce_b_0 = relay.var(
        "inception_4c/5x5_reduce_b_0", shape=(24, ), dtype="float32")
    inception_4c_5x5_w_0 = relay.var(
        "inception_4c/5x5_w_0", shape=(64, 24, 5, 5), dtype="float32")
    inception_4c_5x5_b_0 = relay.var(
        "inception_4c/5x5_b_0", shape=(64, ), dtype="float32")
    inception_4c_pool_proj_w_0 = relay.var(
        "inception_4c/pool_proj_w_0", shape=(64, 512, 1, 1), dtype="float32")
    inception_4c_pool_proj_b_0 = relay.var(
        "inception_4c/pool_proj_b_0", shape=(64, ), dtype="float32")
    inception_4d_1x1_w_0 = relay.var(
        "inception_4d/1x1_w_0", shape=(112, 512, 1, 1), dtype="float32")
    inception_4d_1x1_b_0 = relay.var(
        "inception_4d/1x1_b_0", shape=(112, ), dtype="float32")
    inception_4d_3x3_reduce_w_0 = relay.var(
        "inception_4d/3x3_reduce_w_0", shape=(144, 512, 1, 1), dtype="float32")
    inception_4d_3x3_reduce_b_0 = relay.var(
        "inception_4d/3x3_reduce_b_0", shape=(144, ), dtype="float32")
    inception_4d_3x3_w_0 = relay.var(
        "inception_4d/3x3_w_0", shape=(288, 144, 3, 3), dtype="float32")
    inception_4d_3x3_b_0 = relay.var(
        "inception_4d/3x3_b_0", shape=(288, ), dtype="float32")
    inception_4d_5x5_reduce_w_0 = relay.var(
        "inception_4d/5x5_reduce_w_0", shape=(32, 512, 1, 1), dtype="float32")
    inception_4d_5x5_reduce_b_0 = relay.var(
        "inception_4d/5x5_reduce_b_0", shape=(32, ), dtype="float32")
    inception_4d_5x5_w_0 = relay.var(
        "inception_4d/5x5_w_0", shape=(64, 32, 5, 5), dtype="float32")
    inception_4d_5x5_b_0 = relay.var(
        "inception_4d/5x5_b_0", shape=(64, ), dtype="float32")
    inception_4d_pool_proj_w_0 = relay.var(
        "inception_4d/pool_proj_w_0", shape=(64, 512, 1, 1), dtype="float32")
    inception_4d_pool_proj_b_0 = relay.var(
        "inception_4d/pool_proj_b_0", shape=(64, ), dtype="float32")
    inception_4e_1x1_w_0 = relay.var(
        "inception_4e/1x1_w_0", shape=(256, 528, 1, 1), dtype="float32")
    inception_4e_1x1_b_0 = relay.var(
        "inception_4e/1x1_b_0", shape=(256, ), dtype="float32")
    inception_4e_3x3_reduce_w_0 = relay.var(
        "inception_4e/3x3_reduce_w_0", shape=(160, 528, 1, 1), dtype="float32")
    inception_4e_3x3_reduce_b_0 = relay.var(
        "inception_4e/3x3_reduce_b_0", shape=(160, ), dtype="float32")
    inception_4e_3x3_w_0 = relay.var(
        "inception_4e/3x3_w_0", shape=(320, 160, 3, 3), dtype="float32")
    inception_4e_3x3_b_0 = relay.var(
        "inception_4e/3x3_b_0", shape=(320, ), dtype="float32")
    inception_4e_5x5_reduce_w_0 = relay.var(
        "inception_4e/5x5_reduce_w_0", shape=(32, 528, 1, 1), dtype="float32")
    inception_4e_5x5_reduce_b_0 = relay.var(
        "inception_4e/5x5_reduce_b_0", shape=(32, ), dtype="float32")
    inception_4e_5x5_w_0 = relay.var(
        "inception_4e/5x5_w_0", shape=(128, 32, 5, 5), dtype="float32")
    inception_4e_5x5_b_0 = relay.var(
        "inception_4e/5x5_b_0", shape=(128, ), dtype="float32")
    inception_4e_pool_proj_w_0 = relay.var(
        "inception_4e/pool_proj_w_0", shape=(128, 528, 1, 1), dtype="float32")
    inception_4e_pool_proj_b_0 = relay.var(
        "inception_4e/pool_proj_b_0", shape=(128, ), dtype="float32")
    inception_5a_1x1_w_0 = relay.var(
        "inception_5a/1x1_w_0", shape=(256, 832, 1, 1), dtype="float32")
    inception_5a_1x1_b_0 = relay.var(
        "inception_5a/1x1_b_0", shape=(256, ), dtype="float32")
    inception_5a_3x3_reduce_w_0 = relay.var(
        "inception_5a/3x3_reduce_w_0", shape=(160, 832, 1, 1), dtype="float32")
    inception_5a_3x3_reduce_b_0 = relay.var(
        "inception_5a/3x3_reduce_b_0", shape=(160, ), dtype="float32")
    inception_5a_3x3_w_0 = relay.var(
        "inception_5a/3x3_w_0", shape=(320, 160, 3, 3), dtype="float32")
    inception_5a_3x3_b_0 = relay.var(
        "inception_5a/3x3_b_0", shape=(320, ), dtype="float32")
    inception_5a_5x5_reduce_w_0 = relay.var(
        "inception_5a/5x5_reduce_w_0", shape=(32, 832, 1, 1), dtype="float32")
    inception_5a_5x5_reduce_b_0 = relay.var(
        "inception_5a/5x5_reduce_b_0", shape=(32, ), dtype="float32")
    inception_5a_5x5_w_0 = relay.var(
        "inception_5a/5x5_w_0", shape=(128, 32, 5, 5), dtype="float32")
    inception_5a_5x5_b_0 = relay.var(
        "inception_5a/5x5_b_0", shape=(128, ), dtype="float32")
    inception_5a_pool_proj_w_0 = relay.var(
        "inception_5a/pool_proj_w_0", shape=(128, 832, 1, 1), dtype="float32")
    inception_5a_pool_proj_b_0 = relay.var(
        "inception_5a/pool_proj_b_0", shape=(128, ), dtype="float32")
    inception_5b_1x1_w_0 = relay.var(
        "inception_5b/1x1_w_0", shape=(384, 832, 1, 1), dtype="float32")
    inception_5b_1x1_b_0 = relay.var(
        "inception_5b/1x1_b_0", shape=(384, ), dtype="float32")
    inception_5b_3x3_reduce_w_0 = relay.var(
        "inception_5b/3x3_reduce_w_0", shape=(192, 832, 1, 1), dtype="float32")
    inception_5b_3x3_reduce_b_0 = relay.var(
        "inception_5b/3x3_reduce_b_0", shape=(192, ), dtype="float32")
    inception_5b_3x3_w_0 = relay.var(
        "inception_5b/3x3_w_0", shape=(384, 192, 3, 3), dtype="float32")
    inception_5b_3x3_b_0 = relay.var(
        "inception_5b/3x3_b_0", shape=(384, ), dtype="float32")
    inception_5b_5x5_reduce_w_0 = relay.var(
        "inception_5b/5x5_reduce_w_0", shape=(48, 832, 1, 1), dtype="float32")
    inception_5b_5x5_reduce_b_0 = relay.var(
        "inception_5b/5x5_reduce_b_0", shape=(48, ), dtype="float32")
    inception_5b_5x5_w_0 = relay.var(
        "inception_5b/5x5_w_0", shape=(128, 48, 5, 5), dtype="float32")
    inception_5b_5x5_b_0 = relay.var(
        "inception_5b/5x5_b_0", shape=(128, ), dtype="float32")
    inception_5b_pool_proj_w_0 = relay.var(
        "inception_5b/pool_proj_w_0", shape=(128, 832, 1, 1), dtype="float32")
    inception_5b_pool_proj_b_0 = relay.var(
        "inception_5b/pool_proj_b_0", shape=(128, ), dtype="float32")
    loss3_classifier_w_0 = relay.var(
        "loss3/classifier_w_0", shape=(1000, 1024), dtype="float32")
    loss3_classifier_b_0 = relay.var(
        "loss3/classifier_b_0", shape=(1000, ), dtype="float32")
    OC2_DUMMY_1 = relay.var("OC2_DUMMY_1", shape=(2, ), dtype="float32")

    call_0 = relay.nn.conv2d(data_0, conv1_7x7_s2_w_0, strides=[2, 2], padding=[
                             3, 3, 3, 3], channels=64, kernel_size=[7, 7])
    call_1 = relay.nn.bias_add(call_0, conv1_7x7_s2_b_0)
    call_2 = relay.nn.relu(call_1)
    call_3 = relay.nn.max_pool2d(call_2, pool_size=[3, 3], strides=[
                                 2, 2], padding=[0, 0, 2, 2])
    call_4 = relay.nn.lrn(call_3, bias=1)
    call_5 = relay.nn.conv2d(call_4, conv2_3x3_reduce_w_0, padding=[
                             0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_6 = relay.nn.bias_add(call_5, conv2_3x3_reduce_b_0)
    call_7 = relay.nn.relu(call_6)
    call_8 = relay.nn.conv2d(call_7, conv2_3x3_w_0, padding=[
                             1, 1, 1, 1], channels=192, kernel_size=[3, 3])
    call_9 = relay.nn.bias_add(call_8, conv2_3x3_b_0)
    call_10 = relay.nn.relu(call_9)
    call_11 = relay.nn.lrn(call_10, bias=1)
    call_12 = relay.nn.max_pool2d(call_11, pool_size=[3, 3], strides=[
                                  2, 2], padding=[0, 0, 2, 2])
    call_13 = relay.nn.conv2d(call_12, inception_3a_1x1_w_0, padding=[
                              0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_14 = relay.nn.bias_add(call_13, inception_3a_1x1_b_0)
    call_15 = relay.nn.conv2d(call_12, inception_3a_3x3_reduce_w_0, padding=[
                              0, 0, 0, 0], channels=96, kernel_size=[1, 1])
    call_16 = relay.nn.bias_add(call_15, inception_3a_3x3_reduce_b_0)
    call_17 = relay.nn.relu(call_16)
    call_18 = relay.nn.conv2d(call_17, inception_3a_3x3_w_0, padding=[
                              1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_19 = relay.nn.bias_add(call_18, inception_3a_3x3_b_0)
    call_20 = relay.nn.conv2d(call_12, inception_3a_5x5_reduce_w_0, padding=[
                              0, 0, 0, 0], channels=16, kernel_size=[1, 1])
    call_21 = relay.nn.bias_add(call_20, inception_3a_5x5_reduce_b_0)
    call_22 = relay.nn.relu(call_21)
    call_23 = relay.nn.conv2d(call_22, inception_3a_5x5_w_0, padding=[
                              2, 2, 2, 2], channels=32, kernel_size=[5, 5])
    call_24 = relay.nn.bias_add(call_23, inception_3a_5x5_b_0)
    call_25 = relay.nn.max_pool2d(
        call_12, pool_size=[3, 3], padding=[1, 1, 1, 1])
    call_26 = relay.nn.conv2d(call_25, inception_3a_pool_proj_w_0, padding=[
                              0, 0, 0, 0], channels=32, kernel_size=[1, 1])
    call_27 = relay.nn.bias_add(call_26, inception_3a_pool_proj_b_0)
    call_28 = relay.nn.relu(call_14)
    call_29 = relay.nn.relu(call_19)
    call_30 = relay.nn.relu(call_24)
    call_31 = relay.nn.relu(call_27)
    call_33 = relay.concatenate(relay.Tuple(
        [call_28, call_29, call_30, call_31]), axis=1)
    call_34 = relay.nn.conv2d(call_33, inception_3b_1x1_w_0, padding=[
                              0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_35 = relay.nn.bias_add(call_34, inception_3b_1x1_b_0)
    call_36 = relay.nn.conv2d(call_33, inception_3b_3x3_reduce_w_0, padding=[
                              0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_37 = relay.nn.bias_add(call_36, inception_3b_3x3_reduce_b_0)
    call_38 = relay.nn.relu(call_37)
    call_39 = relay.nn.conv2d(call_38, inception_3b_3x3_w_0, padding=[
                              1, 1, 1, 1], channels=192, kernel_size=[3, 3])
    call_40 = relay.nn.bias_add(call_39, inception_3b_3x3_b_0)
    call_41 = relay.nn.conv2d(call_33, inception_3b_5x5_reduce_w_0, padding=[
                              0, 0, 0, 0], channels=32, kernel_size=[1, 1])
    call_42 = relay.nn.bias_add(call_41, inception_3b_5x5_reduce_b_0)
    call_43 = relay.nn.relu(call_42)
    call_44 = relay.nn.conv2d(call_43, inception_3b_5x5_w_0, padding=[
                              2, 2, 2, 2], channels=96, kernel_size=[5, 5])
    call_45 = relay.nn.bias_add(call_44, inception_3b_5x5_b_0)
    call_46 = relay.nn.max_pool2d(
        call_33, pool_size=[3, 3], padding=[1, 1, 1, 1])
    call_47 = relay.nn.conv2d(call_46, inception_3b_pool_proj_w_0, padding=[
                              0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_48 = relay.nn.bias_add(call_47, inception_3b_pool_proj_b_0)
    call_49 = relay.nn.relu(call_35)
    call_50 = relay.nn.relu(call_40)
    call_51 = relay.nn.relu(call_45)
    call_52 = relay.nn.relu(call_48)
    call_54 = relay.concatenate(relay.Tuple(
        [call_49, call_50, call_51, call_52]), axis=1)
    call_55 = relay.nn.max_pool2d(call_54, pool_size=[3, 3], strides=[
                                  2, 2], padding=[0, 0, 2, 2])
    call_56 = relay.nn.conv2d(call_55, inception_4a_1x1_w_0, padding=[
                              0, 0, 0, 0], channels=192, kernel_size=[1, 1])
    call_57 = relay.nn.bias_add(call_56, inception_4a_1x1_b_0)
    call_58 = relay.nn.conv2d(call_55, inception_4a_3x3_reduce_w_0, padding=[
                              0, 0, 0, 0], channels=96, kernel_size=[1, 1])
    call_59 = relay.nn.bias_add(call_58, inception_4a_3x3_reduce_b_0)
    call_60 = relay.nn.relu(call_59)
    call_61 = relay.nn.conv2d(call_60, inception_4a_3x3_w_0, padding=[
                              1, 1, 1, 1], channels=208, kernel_size=[3, 3])
    call_62 = relay.nn.bias_add(call_61, inception_4a_3x3_b_0)
    call_63 = relay.nn.conv2d(call_55, inception_4a_5x5_reduce_w_0, padding=[
                              0, 0, 0, 0], channels=16, kernel_size=[1, 1])
    call_64 = relay.nn.bias_add(call_63, inception_4a_5x5_reduce_b_0)
    call_65 = relay.nn.relu(call_64)
    call_66 = relay.nn.conv2d(call_65, inception_4a_5x5_w_0, padding=[
                              2, 2, 2, 2], channels=48, kernel_size=[5, 5])
    call_67 = relay.nn.bias_add(call_66, inception_4a_5x5_b_0)
    call_68 = relay.nn.max_pool2d(
        call_55, pool_size=[3, 3], padding=[1, 1, 1, 1])
    call_69 = relay.nn.conv2d(call_68, inception_4a_pool_proj_w_0, padding=[
                              0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_70 = relay.nn.bias_add(call_69, inception_4a_pool_proj_b_0)
    call_71 = relay.nn.relu(call_57)
    call_72 = relay.nn.relu(call_62)
    call_73 = relay.nn.relu(call_67)
    call_74 = relay.nn.relu(call_70)
    call_76 = relay.concatenate(relay.Tuple(
        [call_71, call_72, call_73, call_74]), axis=1)
    call_77 = relay.nn.conv2d(call_76, inception_4b_1x1_w_0, padding=[
                              0, 0, 0, 0], channels=160, kernel_size=[1, 1])
    call_78 = relay.nn.bias_add(call_77, inception_4b_1x1_b_0)
    call_79 = relay.nn.conv2d(call_76, inception_4b_3x3_reduce_w_0, padding=[
                              0, 0, 0, 0], channels=112, kernel_size=[1, 1])
    call_80 = relay.nn.bias_add(call_79, inception_4b_3x3_reduce_b_0)
    call_81 = relay.nn.relu(call_80)
    call_82 = relay.nn.conv2d(call_81, inception_4b_3x3_w_0, padding=[
                              1, 1, 1, 1], channels=224, kernel_size=[3, 3])
    call_83 = relay.nn.bias_add(call_82, inception_4b_3x3_b_0)
    call_84 = relay.nn.conv2d(call_76, inception_4b_5x5_reduce_w_0, padding=[
                              0, 0, 0, 0], channels=24, kernel_size=[1, 1])
    call_85 = relay.nn.bias_add(call_84, inception_4b_5x5_reduce_b_0)
    call_86 = relay.nn.relu(call_85)
    call_87 = relay.nn.conv2d(call_86, inception_4b_5x5_w_0, padding=[
                              2, 2, 2, 2], channels=64, kernel_size=[5, 5])
    call_88 = relay.nn.bias_add(call_87, inception_4b_5x5_b_0)
    call_89 = relay.nn.max_pool2d(
        call_76, pool_size=[3, 3], padding=[1, 1, 1, 1])
    call_90 = relay.nn.conv2d(call_89, inception_4b_pool_proj_w_0, padding=[
                              0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_91 = relay.nn.bias_add(call_90, inception_4b_pool_proj_b_0)
    call_92 = relay.nn.relu(call_78)
    call_93 = relay.nn.relu(call_83)
    call_94 = relay.nn.relu(call_88)
    call_95 = relay.nn.relu(call_91)
    call_97 = relay.concatenate(relay.Tuple(
        [call_92, call_93, call_94, call_95]), axis=1)
    call_98 = relay.nn.conv2d(call_97, inception_4c_1x1_w_0, padding=[
                              0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_99 = relay.nn.bias_add(call_98, inception_4c_1x1_b_0)
    call_100 = relay.nn.conv2d(call_97, inception_4c_3x3_reduce_w_0, padding=[
                               0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_101 = relay.nn.bias_add(call_100, inception_4c_3x3_reduce_b_0)
    call_102 = relay.nn.relu(call_101)
    call_103 = relay.nn.conv2d(call_102, inception_4c_3x3_w_0, padding=[
                               1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_104 = relay.nn.bias_add(call_103, inception_4c_3x3_b_0)
    call_105 = relay.nn.conv2d(call_97, inception_4c_5x5_reduce_w_0, padding=[
                               0, 0, 0, 0], channels=24, kernel_size=[1, 1])
    call_106 = relay.nn.bias_add(call_105, inception_4c_5x5_reduce_b_0)
    call_107 = relay.nn.relu(call_106)
    call_108 = relay.nn.conv2d(call_107, inception_4c_5x5_w_0, padding=[
                               2, 2, 2, 2], channels=64, kernel_size=[5, 5])
    call_109 = relay.nn.bias_add(call_108, inception_4c_5x5_b_0)
    call_110 = relay.nn.max_pool2d(
        call_97, pool_size=[3, 3], padding=[1, 1, 1, 1])
    call_111 = relay.nn.conv2d(call_110, inception_4c_pool_proj_w_0, padding=[
                               0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_112 = relay.nn.bias_add(call_111, inception_4c_pool_proj_b_0)
    call_113 = relay.nn.relu(call_99)
    call_114 = relay.nn.relu(call_104)
    call_115 = relay.nn.relu(call_109)
    call_116 = relay.nn.relu(call_112)
    call_118 = relay.concatenate(relay.Tuple(
        [call_113, call_114, call_115, call_116]), axis=1)
    call_119 = relay.nn.conv2d(call_118, inception_4d_1x1_w_0, padding=[
                               0, 0, 0, 0], channels=112, kernel_size=[1, 1])
    call_120 = relay.nn.bias_add(call_119, inception_4d_1x1_b_0)
    call_121 = relay.nn.conv2d(call_118, inception_4d_3x3_reduce_w_0, padding=[
                               0, 0, 0, 0], channels=144, kernel_size=[1, 1])
    call_122 = relay.nn.bias_add(call_121, inception_4d_3x3_reduce_b_0)
    call_123 = relay.nn.relu(call_122)
    call_124 = relay.nn.conv2d(call_123, inception_4d_3x3_w_0, padding=[
                               1, 1, 1, 1], channels=288, kernel_size=[3, 3])
    call_125 = relay.nn.bias_add(call_124, inception_4d_3x3_b_0)
    call_126 = relay.nn.conv2d(call_118, inception_4d_5x5_reduce_w_0, padding=[
                               0, 0, 0, 0], channels=32, kernel_size=[1, 1])
    call_127 = relay.nn.bias_add(call_126, inception_4d_5x5_reduce_b_0)
    call_128 = relay.nn.relu(call_127)
    call_129 = relay.nn.conv2d(call_128, inception_4d_5x5_w_0, padding=[
                               2, 2, 2, 2], channels=64, kernel_size=[5, 5])
    call_130 = relay.nn.bias_add(call_129, inception_4d_5x5_b_0)
    call_131 = relay.nn.max_pool2d(
        call_118, pool_size=[3, 3], padding=[1, 1, 1, 1])
    call_132 = relay.nn.conv2d(call_131, inception_4d_pool_proj_w_0, padding=[
                               0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_133 = relay.nn.bias_add(call_132, inception_4d_pool_proj_b_0)
    call_134 = relay.nn.relu(call_120)
    call_135 = relay.nn.relu(call_125)
    call_136 = relay.nn.relu(call_130)
    call_137 = relay.nn.relu(call_133)
    call_139 = relay.concatenate(relay.Tuple(
        [call_134, call_135, call_136, call_137]), axis=1)
    call_140 = relay.nn.conv2d(call_139, inception_4e_1x1_w_0, padding=[
                               0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_141 = relay.nn.bias_add(call_140, inception_4e_1x1_b_0)
    call_142 = relay.nn.conv2d(call_139, inception_4e_3x3_reduce_w_0, padding=[
                               0, 0, 0, 0], channels=160, kernel_size=[1, 1])
    call_143 = relay.nn.bias_add(call_142, inception_4e_3x3_reduce_b_0)
    call_144 = relay.nn.relu(call_143)
    call_145 = relay.nn.conv2d(call_144, inception_4e_3x3_w_0, padding=[
                               1, 1, 1, 1], channels=320, kernel_size=[3, 3])
    call_146 = relay.nn.bias_add(call_145, inception_4e_3x3_b_0)
    call_147 = relay.nn.conv2d(call_139, inception_4e_5x5_reduce_w_0, padding=[
                               0, 0, 0, 0], channels=32, kernel_size=[1, 1])
    call_148 = relay.nn.bias_add(call_147, inception_4e_5x5_reduce_b_0)
    call_149 = relay.nn.relu(call_148)
    call_150 = relay.nn.conv2d(call_149, inception_4e_5x5_w_0, padding=[
                               2, 2, 2, 2], channels=128, kernel_size=[5, 5])
    call_151 = relay.nn.bias_add(call_150, inception_4e_5x5_b_0)
    call_152 = relay.nn.max_pool2d(
        call_139, pool_size=[3, 3], padding=[1, 1, 1, 1])
    call_153 = relay.nn.conv2d(call_152, inception_4e_pool_proj_w_0, padding=[
                               0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_154 = relay.nn.bias_add(call_153, inception_4e_pool_proj_b_0)
    call_155 = relay.nn.relu(call_141)
    call_156 = relay.nn.relu(call_146)
    call_157 = relay.nn.relu(call_151)
    call_158 = relay.nn.relu(call_154)
    call_160 = relay.concatenate(relay.Tuple(
        [call_155, call_156, call_157, call_158]), axis=1)
    call_161 = relay.nn.max_pool2d(call_160, pool_size=[3, 3], strides=[
                                   2, 2], padding=[0, 0, 2, 2])
    call_162 = relay.nn.conv2d(call_161, inception_5a_1x1_w_0, padding=[
                               0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_163 = relay.nn.bias_add(call_162, inception_5a_1x1_b_0)
    call_164 = relay.nn.conv2d(call_161, inception_5a_3x3_reduce_w_0, padding=[
                               0, 0, 0, 0], channels=160, kernel_size=[1, 1])
    call_165 = relay.nn.bias_add(call_164, inception_5a_3x3_reduce_b_0)
    call_166 = relay.nn.relu(call_165)
    call_167 = relay.nn.conv2d(call_166, inception_5a_3x3_w_0, padding=[
                               1, 1, 1, 1], channels=320, kernel_size=[3, 3])
    call_168 = relay.nn.bias_add(call_167, inception_5a_3x3_b_0)
    call_169 = relay.nn.conv2d(call_161, inception_5a_5x5_reduce_w_0, padding=[
                               0, 0, 0, 0], channels=32, kernel_size=[1, 1])
    call_170 = relay.nn.bias_add(call_169, inception_5a_5x5_reduce_b_0)
    call_171 = relay.nn.relu(call_170)
    call_172 = relay.nn.conv2d(call_171, inception_5a_5x5_w_0, padding=[
                               2, 2, 2, 2], channels=128, kernel_size=[5, 5])
    call_173 = relay.nn.bias_add(call_172, inception_5a_5x5_b_0)
    call_174 = relay.nn.max_pool2d(
        call_161, pool_size=[3, 3], padding=[1, 1, 1, 1])
    call_175 = relay.nn.conv2d(call_174, inception_5a_pool_proj_w_0, padding=[
                               0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_176 = relay.nn.bias_add(call_175, inception_5a_pool_proj_b_0)
    call_177 = relay.nn.relu(call_163)
    call_178 = relay.nn.relu(call_168)
    call_179 = relay.nn.relu(call_173)
    call_180 = relay.nn.relu(call_176)
    call_182 = relay.concatenate(relay.Tuple(
        [call_177, call_178, call_179, call_180]), axis=1)
    call_183 = relay.nn.conv2d(call_182, inception_5b_1x1_w_0, padding=[
                               0, 0, 0, 0], channels=384, kernel_size=[1, 1])
    call_184 = relay.nn.bias_add(call_183, inception_5b_1x1_b_0)
    call_185 = relay.nn.conv2d(call_182, inception_5b_3x3_reduce_w_0, padding=[
                               0, 0, 0, 0], channels=192, kernel_size=[1, 1])
    call_186 = relay.nn.bias_add(call_185, inception_5b_3x3_reduce_b_0)
    call_187 = relay.nn.relu(call_186)
    call_188 = relay.nn.conv2d(call_187, inception_5b_3x3_w_0, padding=[
                               1, 1, 1, 1], channels=384, kernel_size=[3, 3])
    call_189 = relay.nn.bias_add(call_188, inception_5b_3x3_b_0)
    call_190 = relay.nn.conv2d(call_182, inception_5b_5x5_reduce_w_0, padding=[
                               0, 0, 0, 0], channels=48, kernel_size=[1, 1])
    call_191 = relay.nn.bias_add(call_190, inception_5b_5x5_reduce_b_0)
    call_192 = relay.nn.relu(call_191)
    call_193 = relay.nn.conv2d(call_192, inception_5b_5x5_w_0, padding=[
                               2, 2, 2, 2], channels=128, kernel_size=[5, 5])
    call_194 = relay.nn.bias_add(call_193, inception_5b_5x5_b_0)
    call_195 = relay.nn.max_pool2d(
        call_182, pool_size=[3, 3], padding=[1, 1, 1, 1])
    call_196 = relay.nn.conv2d(call_195, inception_5b_pool_proj_w_0, padding=[
                               0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_197 = relay.nn.bias_add(call_196, inception_5b_pool_proj_b_0)
    call_198 = relay.nn.relu(call_184)
    call_199 = relay.nn.relu(call_189)
    call_200 = relay.nn.relu(call_194)
    call_201 = relay.nn.relu(call_197)
    call_203 = relay.concatenate(relay.Tuple(
        [call_198, call_199, call_200, call_201]), axis=1)
    call_204 = relay.nn.avg_pool2d(
        call_203, pool_size=[7, 7], padding=[0, 0, 0, 0])
    call_205_0 = relay.nn.dropout(call_204, rate=0.4)
    call_207 = relay.reshape(call_205_0, newshape=[1, 1024])
    call_208 = relay.nn.batch_flatten(call_207)
    call_209 = relay.nn.dense(call_208, loss3_classifier_w_0, units=1000)
    call_210 = relay.multiply(relay.const(
        1.0, dtype="float32"), loss3_classifier_b_0)
    call_211 = relay.add(call_209, call_210)
    call_212 = relay.max(call_211, axis=[1], keepdims=True)
    call_213 = relay.subtract(call_211, call_212)
    call_214 = relay.exp(call_213)
    call_215 = relay.sum(call_214, axis=[1], keepdims=True)
    call_output0 = relay.divide(call_214, call_215)
    return call_output0
