from tvm import relay, IRModule
import numpy as np


def ResnetModule():
    data = relay.var("data", shape=(1, 3, 224, 224), dtype="float32")
    resnetv24_batchnorm0_gamma = relay.var(
        "resnetv24_batchnorm0_gamma", shape=(3, ), dtype="float32")
    resnetv24_batchnorm0_beta = relay.var(
        "resnetv24_batchnorm0_beta", shape=(3, ), dtype="float32")
    resnetv24_batchnorm0_running_mean = relay.var(
        "resnetv24_batchnorm0_running_mean", shape=(3, ), dtype="float32")
    resnetv24_batchnorm0_running_var = relay.var(
        "resnetv24_batchnorm0_running_var", shape=(3, ), dtype="float32")
    resnetv24_conv0_weight = relay.var(
        "resnetv24_conv0_weight", shape=(64, 3, 7, 7), dtype="float32")
    resnetv24_batchnorm1_gamma = relay.var(
        "resnetv24_batchnorm1_gamma", shape=(64, ), dtype="float32")
    resnetv24_batchnorm1_beta = relay.var(
        "resnetv24_batchnorm1_beta", shape=(64, ), dtype="float32")
    resnetv24_batchnorm1_running_mean = relay.var(
        "resnetv24_batchnorm1_running_mean", shape=(64, ), dtype="float32")
    resnetv24_batchnorm1_running_var = relay.var(
        "resnetv24_batchnorm1_running_var", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm0_gamma = relay.var(
        "resnetv24_stage1_batchnorm0_gamma", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm0_beta = relay.var(
        "resnetv24_stage1_batchnorm0_beta", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm0_running_mean = relay.var(
        "resnetv24_stage1_batchnorm0_running_mean", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm0_running_var = relay.var(
        "resnetv24_stage1_batchnorm0_running_var", shape=(64, ), dtype="float32")
    resnetv24_stage1_conv0_weight = relay.var(
        "resnetv24_stage1_conv0_weight", shape=(64, 64, 1, 1), dtype="float32")
    resnetv24_stage1_batchnorm1_gamma = relay.var(
        "resnetv24_stage1_batchnorm1_gamma", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm1_beta = relay.var(
        "resnetv24_stage1_batchnorm1_beta", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm1_running_mean = relay.var(
        "resnetv24_stage1_batchnorm1_running_mean", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm1_running_var = relay.var(
        "resnetv24_stage1_batchnorm1_running_var", shape=(64, ), dtype="float32")
    resnetv24_stage1_conv1_weight = relay.var(
        "resnetv24_stage1_conv1_weight", shape=(64, 64, 3, 3), dtype="float32")
    resnetv24_stage1_batchnorm2_gamma = relay.var(
        "resnetv24_stage1_batchnorm2_gamma", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm2_beta = relay.var(
        "resnetv24_stage1_batchnorm2_beta", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm2_running_mean = relay.var(
        "resnetv24_stage1_batchnorm2_running_mean", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm2_running_var = relay.var(
        "resnetv24_stage1_batchnorm2_running_var", shape=(64, ), dtype="float32")
    resnetv24_stage1_conv2_weight = relay.var(
        "resnetv24_stage1_conv2_weight", shape=(256, 64, 1, 1), dtype="float32")
    resnetv24_stage1_conv3_weight = relay.var(
        "resnetv24_stage1_conv3_weight", shape=(256, 64, 1, 1), dtype="float32")
    resnetv24_stage1_batchnorm3_gamma = relay.var(
        "resnetv24_stage1_batchnorm3_gamma", shape=(256, ), dtype="float32")
    resnetv24_stage1_batchnorm3_beta = relay.var(
        "resnetv24_stage1_batchnorm3_beta", shape=(256, ), dtype="float32")
    resnetv24_stage1_batchnorm3_running_mean = relay.var(
        "resnetv24_stage1_batchnorm3_running_mean", shape=(256, ), dtype="float32")
    resnetv24_stage1_batchnorm3_running_var = relay.var(
        "resnetv24_stage1_batchnorm3_running_var", shape=(256, ), dtype="float32")
    resnetv24_stage1_conv4_weight = relay.var(
        "resnetv24_stage1_conv4_weight", shape=(64, 256, 1, 1), dtype="float32")
    resnetv24_stage1_batchnorm4_gamma = relay.var(
        "resnetv24_stage1_batchnorm4_gamma", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm4_beta = relay.var(
        "resnetv24_stage1_batchnorm4_beta", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm4_running_mean = relay.var(
        "resnetv24_stage1_batchnorm4_running_mean", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm4_running_var = relay.var(
        "resnetv24_stage1_batchnorm4_running_var", shape=(64, ), dtype="float32")
    resnetv24_stage1_conv5_weight = relay.var(
        "resnetv24_stage1_conv5_weight", shape=(64, 64, 3, 3), dtype="float32")
    resnetv24_stage1_batchnorm5_gamma = relay.var(
        "resnetv24_stage1_batchnorm5_gamma", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm5_beta = relay.var(
        "resnetv24_stage1_batchnorm5_beta", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm5_running_mean = relay.var(
        "resnetv24_stage1_batchnorm5_running_mean", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm5_running_var = relay.var(
        "resnetv24_stage1_batchnorm5_running_var", shape=(64, ), dtype="float32")
    resnetv24_stage1_conv6_weight = relay.var(
        "resnetv24_stage1_conv6_weight", shape=(256, 64, 1, 1), dtype="float32")
    resnetv24_stage1_batchnorm6_gamma = relay.var(
        "resnetv24_stage1_batchnorm6_gamma", shape=(256, ), dtype="float32")
    resnetv24_stage1_batchnorm6_beta = relay.var(
        "resnetv24_stage1_batchnorm6_beta", shape=(256, ), dtype="float32")
    resnetv24_stage1_batchnorm6_running_mean = relay.var(
        "resnetv24_stage1_batchnorm6_running_mean", shape=(256, ), dtype="float32")
    resnetv24_stage1_batchnorm6_running_var = relay.var(
        "resnetv24_stage1_batchnorm6_running_var", shape=(256, ), dtype="float32")
    resnetv24_stage1_conv7_weight = relay.var(
        "resnetv24_stage1_conv7_weight", shape=(64, 256, 1, 1), dtype="float32")
    resnetv24_stage1_batchnorm7_gamma = relay.var(
        "resnetv24_stage1_batchnorm7_gamma", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm7_beta = relay.var(
        "resnetv24_stage1_batchnorm7_beta", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm7_running_mean = relay.var(
        "resnetv24_stage1_batchnorm7_running_mean", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm7_running_var = relay.var(
        "resnetv24_stage1_batchnorm7_running_var", shape=(64, ), dtype="float32")
    resnetv24_stage1_conv8_weight = relay.var(
        "resnetv24_stage1_conv8_weight", shape=(64, 64, 3, 3), dtype="float32")
    resnetv24_stage1_batchnorm8_gamma = relay.var(
        "resnetv24_stage1_batchnorm8_gamma", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm8_beta = relay.var(
        "resnetv24_stage1_batchnorm8_beta", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm8_running_mean = relay.var(
        "resnetv24_stage1_batchnorm8_running_mean", shape=(64, ), dtype="float32")
    resnetv24_stage1_batchnorm8_running_var = relay.var(
        "resnetv24_stage1_batchnorm8_running_var", shape=(64, ), dtype="float32")
    resnetv24_stage1_conv9_weight = relay.var(
        "resnetv24_stage1_conv9_weight", shape=(256, 64, 1, 1), dtype="float32")
    resnetv24_stage2_batchnorm0_gamma = relay.var(
        "resnetv24_stage2_batchnorm0_gamma", shape=(256, ), dtype="float32")
    resnetv24_stage2_batchnorm0_beta = relay.var(
        "resnetv24_stage2_batchnorm0_beta", shape=(256, ), dtype="float32")
    resnetv24_stage2_batchnorm0_running_mean = relay.var(
        "resnetv24_stage2_batchnorm0_running_mean", shape=(256, ), dtype="float32")
    resnetv24_stage2_batchnorm0_running_var = relay.var(
        "resnetv24_stage2_batchnorm0_running_var", shape=(256, ), dtype="float32")
    resnetv24_stage2_conv0_weight = relay.var(
        "resnetv24_stage2_conv0_weight", shape=(128, 256, 1, 1), dtype="float32")
    resnetv24_stage2_batchnorm1_gamma = relay.var(
        "resnetv24_stage2_batchnorm1_gamma", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm1_beta = relay.var(
        "resnetv24_stage2_batchnorm1_beta", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm1_running_mean = relay.var(
        "resnetv24_stage2_batchnorm1_running_mean", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm1_running_var = relay.var(
        "resnetv24_stage2_batchnorm1_running_var", shape=(128, ), dtype="float32")
    resnetv24_stage2_conv1_weight = relay.var(
        "resnetv24_stage2_conv1_weight", shape=(128, 128, 3, 3), dtype="float32")
    resnetv24_stage2_batchnorm2_gamma = relay.var(
        "resnetv24_stage2_batchnorm2_gamma", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm2_beta = relay.var(
        "resnetv24_stage2_batchnorm2_beta", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm2_running_mean = relay.var(
        "resnetv24_stage2_batchnorm2_running_mean", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm2_running_var = relay.var(
        "resnetv24_stage2_batchnorm2_running_var", shape=(128, ), dtype="float32")
    resnetv24_stage2_conv2_weight = relay.var(
        "resnetv24_stage2_conv2_weight", shape=(512, 128, 1, 1), dtype="float32")
    resnetv24_stage2_conv3_weight = relay.var(
        "resnetv24_stage2_conv3_weight", shape=(512, 256, 1, 1), dtype="float32")
    resnetv24_stage2_batchnorm3_gamma = relay.var(
        "resnetv24_stage2_batchnorm3_gamma", shape=(512, ), dtype="float32")
    resnetv24_stage2_batchnorm3_beta = relay.var(
        "resnetv24_stage2_batchnorm3_beta", shape=(512, ), dtype="float32")
    resnetv24_stage2_batchnorm3_running_mean = relay.var(
        "resnetv24_stage2_batchnorm3_running_mean", shape=(512, ), dtype="float32")
    resnetv24_stage2_batchnorm3_running_var = relay.var(
        "resnetv24_stage2_batchnorm3_running_var", shape=(512, ), dtype="float32")
    resnetv24_stage2_conv4_weight = relay.var(
        "resnetv24_stage2_conv4_weight", shape=(128, 512, 1, 1), dtype="float32")
    resnetv24_stage2_batchnorm4_gamma = relay.var(
        "resnetv24_stage2_batchnorm4_gamma", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm4_beta = relay.var(
        "resnetv24_stage2_batchnorm4_beta", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm4_running_mean = relay.var(
        "resnetv24_stage2_batchnorm4_running_mean", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm4_running_var = relay.var(
        "resnetv24_stage2_batchnorm4_running_var", shape=(128, ), dtype="float32")
    resnetv24_stage2_conv5_weight = relay.var(
        "resnetv24_stage2_conv5_weight", shape=(128, 128, 3, 3), dtype="float32")
    resnetv24_stage2_batchnorm5_gamma = relay.var(
        "resnetv24_stage2_batchnorm5_gamma", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm5_beta = relay.var(
        "resnetv24_stage2_batchnorm5_beta", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm5_running_mean = relay.var(
        "resnetv24_stage2_batchnorm5_running_mean", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm5_running_var = relay.var(
        "resnetv24_stage2_batchnorm5_running_var", shape=(128, ), dtype="float32")
    resnetv24_stage2_conv6_weight = relay.var(
        "resnetv24_stage2_conv6_weight", shape=(512, 128, 1, 1), dtype="float32")
    resnetv24_stage2_batchnorm6_gamma = relay.var(
        "resnetv24_stage2_batchnorm6_gamma", shape=(512, ), dtype="float32")
    resnetv24_stage2_batchnorm6_beta = relay.var(
        "resnetv24_stage2_batchnorm6_beta", shape=(512, ), dtype="float32")
    resnetv24_stage2_batchnorm6_running_mean = relay.var(
        "resnetv24_stage2_batchnorm6_running_mean", shape=(512, ), dtype="float32")
    resnetv24_stage2_batchnorm6_running_var = relay.var(
        "resnetv24_stage2_batchnorm6_running_var", shape=(512, ), dtype="float32")
    resnetv24_stage2_conv7_weight = relay.var(
        "resnetv24_stage2_conv7_weight", shape=(128, 512, 1, 1), dtype="float32")
    resnetv24_stage2_batchnorm7_gamma = relay.var(
        "resnetv24_stage2_batchnorm7_gamma", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm7_beta = relay.var(
        "resnetv24_stage2_batchnorm7_beta", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm7_running_mean = relay.var(
        "resnetv24_stage2_batchnorm7_running_mean", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm7_running_var = relay.var(
        "resnetv24_stage2_batchnorm7_running_var", shape=(128, ), dtype="float32")
    resnetv24_stage2_conv8_weight = relay.var(
        "resnetv24_stage2_conv8_weight", shape=(128, 128, 3, 3), dtype="float32")
    resnetv24_stage2_batchnorm8_gamma = relay.var(
        "resnetv24_stage2_batchnorm8_gamma", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm8_beta = relay.var(
        "resnetv24_stage2_batchnorm8_beta", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm8_running_mean = relay.var(
        "resnetv24_stage2_batchnorm8_running_mean", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm8_running_var = relay.var(
        "resnetv24_stage2_batchnorm8_running_var", shape=(128, ), dtype="float32")
    resnetv24_stage2_conv9_weight = relay.var(
        "resnetv24_stage2_conv9_weight", shape=(512, 128, 1, 1), dtype="float32")
    resnetv24_stage2_batchnorm9_gamma = relay.var(
        "resnetv24_stage2_batchnorm9_gamma", shape=(512, ), dtype="float32")
    resnetv24_stage2_batchnorm9_beta = relay.var(
        "resnetv24_stage2_batchnorm9_beta", shape=(512, ), dtype="float32")
    resnetv24_stage2_batchnorm9_running_mean = relay.var(
        "resnetv24_stage2_batchnorm9_running_mean", shape=(512, ), dtype="float32")
    resnetv24_stage2_batchnorm9_running_var = relay.var(
        "resnetv24_stage2_batchnorm9_running_var", shape=(512, ), dtype="float32")
    resnetv24_stage2_conv10_weight = relay.var(
        "resnetv24_stage2_conv10_weight", shape=(128, 512, 1, 1), dtype="float32")
    resnetv24_stage2_batchnorm10_gamma = relay.var(
        "resnetv24_stage2_batchnorm10_gamma", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm10_beta = relay.var(
        "resnetv24_stage2_batchnorm10_beta", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm10_running_mean = relay.var(
        "resnetv24_stage2_batchnorm10_running_mean", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm10_running_var = relay.var(
        "resnetv24_stage2_batchnorm10_running_var", shape=(128, ), dtype="float32")
    resnetv24_stage2_conv11_weight = relay.var(
        "resnetv24_stage2_conv11_weight", shape=(128, 128, 3, 3), dtype="float32")
    resnetv24_stage2_batchnorm11_gamma = relay.var(
        "resnetv24_stage2_batchnorm11_gamma", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm11_beta = relay.var(
        "resnetv24_stage2_batchnorm11_beta", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm11_running_mean = relay.var(
        "resnetv24_stage2_batchnorm11_running_mean", shape=(128, ), dtype="float32")
    resnetv24_stage2_batchnorm11_running_var = relay.var(
        "resnetv24_stage2_batchnorm11_running_var", shape=(128, ), dtype="float32")
    resnetv24_stage2_conv12_weight = relay.var(
        "resnetv24_stage2_conv12_weight", shape=(512, 128, 1, 1), dtype="float32")
    resnetv24_stage3_batchnorm0_gamma = relay.var(
        "resnetv24_stage3_batchnorm0_gamma", shape=(512, ), dtype="float32")
    resnetv24_stage3_batchnorm0_beta = relay.var(
        "resnetv24_stage3_batchnorm0_beta", shape=(512, ), dtype="float32")
    resnetv24_stage3_batchnorm0_running_mean = relay.var(
        "resnetv24_stage3_batchnorm0_running_mean", shape=(512, ), dtype="float32")
    resnetv24_stage3_batchnorm0_running_var = relay.var(
        "resnetv24_stage3_batchnorm0_running_var", shape=(512, ), dtype="float32")
    resnetv24_stage3_conv0_weight = relay.var(
        "resnetv24_stage3_conv0_weight", shape=(256, 512, 1, 1), dtype="float32")
    resnetv24_stage3_batchnorm1_gamma = relay.var(
        "resnetv24_stage3_batchnorm1_gamma", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm1_beta = relay.var(
        "resnetv24_stage3_batchnorm1_beta", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm1_running_mean = relay.var(
        "resnetv24_stage3_batchnorm1_running_mean", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm1_running_var = relay.var(
        "resnetv24_stage3_batchnorm1_running_var", shape=(256, ), dtype="float32")
    resnetv24_stage3_conv1_weight = relay.var(
        "resnetv24_stage3_conv1_weight", shape=(256, 256, 3, 3), dtype="float32")
    resnetv24_stage3_batchnorm2_gamma = relay.var(
        "resnetv24_stage3_batchnorm2_gamma", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm2_beta = relay.var(
        "resnetv24_stage3_batchnorm2_beta", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm2_running_mean = relay.var(
        "resnetv24_stage3_batchnorm2_running_mean", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm2_running_var = relay.var(
        "resnetv24_stage3_batchnorm2_running_var", shape=(256, ), dtype="float32")
    resnetv24_stage3_conv2_weight = relay.var(
        "resnetv24_stage3_conv2_weight", shape=(1024, 256, 1, 1), dtype="float32")
    resnetv24_stage3_conv3_weight = relay.var(
        "resnetv24_stage3_conv3_weight", shape=(1024, 512, 1, 1), dtype="float32")
    resnetv24_stage3_batchnorm3_gamma = relay.var(
        "resnetv24_stage3_batchnorm3_gamma", shape=(1024, ), dtype="float32")
    resnetv24_stage3_batchnorm3_beta = relay.var(
        "resnetv24_stage3_batchnorm3_beta", shape=(1024, ), dtype="float32")
    resnetv24_stage3_batchnorm3_running_mean = relay.var(
        "resnetv24_stage3_batchnorm3_running_mean", shape=(1024, ), dtype="float32")
    resnetv24_stage3_batchnorm3_running_var = relay.var(
        "resnetv24_stage3_batchnorm3_running_var", shape=(1024, ), dtype="float32")
    resnetv24_stage3_conv4_weight = relay.var(
        "resnetv24_stage3_conv4_weight", shape=(256, 1024, 1, 1), dtype="float32")
    resnetv24_stage3_batchnorm4_gamma = relay.var(
        "resnetv24_stage3_batchnorm4_gamma", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm4_beta = relay.var(
        "resnetv24_stage3_batchnorm4_beta", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm4_running_mean = relay.var(
        "resnetv24_stage3_batchnorm4_running_mean", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm4_running_var = relay.var(
        "resnetv24_stage3_batchnorm4_running_var", shape=(256, ), dtype="float32")
    resnetv24_stage3_conv5_weight = relay.var(
        "resnetv24_stage3_conv5_weight", shape=(256, 256, 3, 3), dtype="float32")
    resnetv24_stage3_batchnorm5_gamma = relay.var(
        "resnetv24_stage3_batchnorm5_gamma", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm5_beta = relay.var(
        "resnetv24_stage3_batchnorm5_beta", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm5_running_mean = relay.var(
        "resnetv24_stage3_batchnorm5_running_mean", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm5_running_var = relay.var(
        "resnetv24_stage3_batchnorm5_running_var", shape=(256, ), dtype="float32")
    resnetv24_stage3_conv6_weight = relay.var(
        "resnetv24_stage3_conv6_weight", shape=(1024, 256, 1, 1), dtype="float32")
    resnetv24_stage3_batchnorm6_gamma = relay.var(
        "resnetv24_stage3_batchnorm6_gamma", shape=(1024, ), dtype="float32")
    resnetv24_stage3_batchnorm6_beta = relay.var(
        "resnetv24_stage3_batchnorm6_beta", shape=(1024, ), dtype="float32")
    resnetv24_stage3_batchnorm6_running_mean = relay.var(
        "resnetv24_stage3_batchnorm6_running_mean", shape=(1024, ), dtype="float32")
    resnetv24_stage3_batchnorm6_running_var = relay.var(
        "resnetv24_stage3_batchnorm6_running_var", shape=(1024, ), dtype="float32")
    resnetv24_stage3_conv7_weight = relay.var(
        "resnetv24_stage3_conv7_weight", shape=(256, 1024, 1, 1), dtype="float32")
    resnetv24_stage3_batchnorm7_gamma = relay.var(
        "resnetv24_stage3_batchnorm7_gamma", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm7_beta = relay.var(
        "resnetv24_stage3_batchnorm7_beta", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm7_running_mean = relay.var(
        "resnetv24_stage3_batchnorm7_running_mean", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm7_running_var = relay.var(
        "resnetv24_stage3_batchnorm7_running_var", shape=(256, ), dtype="float32")
    resnetv24_stage3_conv8_weight = relay.var(
        "resnetv24_stage3_conv8_weight", shape=(256, 256, 3, 3), dtype="float32")
    resnetv24_stage3_batchnorm8_gamma = relay.var(
        "resnetv24_stage3_batchnorm8_gamma", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm8_beta = relay.var(
        "resnetv24_stage3_batchnorm8_beta", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm8_running_mean = relay.var(
        "resnetv24_stage3_batchnorm8_running_mean", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm8_running_var = relay.var(
        "resnetv24_stage3_batchnorm8_running_var", shape=(256, ), dtype="float32")
    resnetv24_stage3_conv9_weight = relay.var(
        "resnetv24_stage3_conv9_weight", shape=(1024, 256, 1, 1), dtype="float32")
    resnetv24_stage3_batchnorm9_gamma = relay.var(
        "resnetv24_stage3_batchnorm9_gamma", shape=(1024, ), dtype="float32")
    resnetv24_stage3_batchnorm9_beta = relay.var(
        "resnetv24_stage3_batchnorm9_beta", shape=(1024, ), dtype="float32")
    resnetv24_stage3_batchnorm9_running_mean = relay.var(
        "resnetv24_stage3_batchnorm9_running_mean", shape=(1024, ), dtype="float32")
    resnetv24_stage3_batchnorm9_running_var = relay.var(
        "resnetv24_stage3_batchnorm9_running_var", shape=(1024, ), dtype="float32")
    resnetv24_stage3_conv10_weight = relay.var(
        "resnetv24_stage3_conv10_weight", shape=(256, 1024, 1, 1), dtype="float32")
    resnetv24_stage3_batchnorm10_gamma = relay.var(
        "resnetv24_stage3_batchnorm10_gamma", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm10_beta = relay.var(
        "resnetv24_stage3_batchnorm10_beta", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm10_running_mean = relay.var(
        "resnetv24_stage3_batchnorm10_running_mean", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm10_running_var = relay.var(
        "resnetv24_stage3_batchnorm10_running_var", shape=(256, ), dtype="float32")
    resnetv24_stage3_conv11_weight = relay.var(
        "resnetv24_stage3_conv11_weight", shape=(256, 256, 3, 3), dtype="float32")
    resnetv24_stage3_batchnorm11_gamma = relay.var(
        "resnetv24_stage3_batchnorm11_gamma", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm11_beta = relay.var(
        "resnetv24_stage3_batchnorm11_beta", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm11_running_mean = relay.var(
        "resnetv24_stage3_batchnorm11_running_mean", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm11_running_var = relay.var(
        "resnetv24_stage3_batchnorm11_running_var", shape=(256, ), dtype="float32")
    resnetv24_stage3_conv12_weight = relay.var(
        "resnetv24_stage3_conv12_weight", shape=(1024, 256, 1, 1), dtype="float32")
    resnetv24_stage3_batchnorm12_gamma = relay.var(
        "resnetv24_stage3_batchnorm12_gamma", shape=(1024, ), dtype="float32")
    resnetv24_stage3_batchnorm12_beta = relay.var(
        "resnetv24_stage3_batchnorm12_beta", shape=(1024, ), dtype="float32")
    resnetv24_stage3_batchnorm12_running_mean = relay.var(
        "resnetv24_stage3_batchnorm12_running_mean", shape=(1024, ), dtype="float32")
    resnetv24_stage3_batchnorm12_running_var = relay.var(
        "resnetv24_stage3_batchnorm12_running_var", shape=(1024, ), dtype="float32")
    resnetv24_stage3_conv13_weight = relay.var(
        "resnetv24_stage3_conv13_weight", shape=(256, 1024, 1, 1), dtype="float32")
    resnetv24_stage3_batchnorm13_gamma = relay.var(
        "resnetv24_stage3_batchnorm13_gamma", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm13_beta = relay.var(
        "resnetv24_stage3_batchnorm13_beta", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm13_running_mean = relay.var(
        "resnetv24_stage3_batchnorm13_running_mean", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm13_running_var = relay.var(
        "resnetv24_stage3_batchnorm13_running_var", shape=(256, ), dtype="float32")
    resnetv24_stage3_conv14_weight = relay.var(
        "resnetv24_stage3_conv14_weight", shape=(256, 256, 3, 3), dtype="float32")
    resnetv24_stage3_batchnorm14_gamma = relay.var(
        "resnetv24_stage3_batchnorm14_gamma", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm14_beta = relay.var(
        "resnetv24_stage3_batchnorm14_beta", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm14_running_mean = relay.var(
        "resnetv24_stage3_batchnorm14_running_mean", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm14_running_var = relay.var(
        "resnetv24_stage3_batchnorm14_running_var", shape=(256, ), dtype="float32")
    resnetv24_stage3_conv15_weight = relay.var(
        "resnetv24_stage3_conv15_weight", shape=(1024, 256, 1, 1), dtype="float32")
    resnetv24_stage3_batchnorm15_gamma = relay.var(
        "resnetv24_stage3_batchnorm15_gamma", shape=(1024, ), dtype="float32")
    resnetv24_stage3_batchnorm15_beta = relay.var(
        "resnetv24_stage3_batchnorm15_beta", shape=(1024, ), dtype="float32")
    resnetv24_stage3_batchnorm15_running_mean = relay.var(
        "resnetv24_stage3_batchnorm15_running_mean", shape=(1024, ), dtype="float32")
    resnetv24_stage3_batchnorm15_running_var = relay.var(
        "resnetv24_stage3_batchnorm15_running_var", shape=(1024, ), dtype="float32")
    resnetv24_stage3_conv16_weight = relay.var(
        "resnetv24_stage3_conv16_weight", shape=(256, 1024, 1, 1), dtype="float32")
    resnetv24_stage3_batchnorm16_gamma = relay.var(
        "resnetv24_stage3_batchnorm16_gamma", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm16_beta = relay.var(
        "resnetv24_stage3_batchnorm16_beta", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm16_running_mean = relay.var(
        "resnetv24_stage3_batchnorm16_running_mean", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm16_running_var = relay.var(
        "resnetv24_stage3_batchnorm16_running_var", shape=(256, ), dtype="float32")
    resnetv24_stage3_conv17_weight = relay.var(
        "resnetv24_stage3_conv17_weight", shape=(256, 256, 3, 3), dtype="float32")
    resnetv24_stage3_batchnorm17_gamma = relay.var(
        "resnetv24_stage3_batchnorm17_gamma", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm17_beta = relay.var(
        "resnetv24_stage3_batchnorm17_beta", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm17_running_mean = relay.var(
        "resnetv24_stage3_batchnorm17_running_mean", shape=(256, ), dtype="float32")
    resnetv24_stage3_batchnorm17_running_var = relay.var(
        "resnetv24_stage3_batchnorm17_running_var", shape=(256, ), dtype="float32")
    resnetv24_stage3_conv18_weight = relay.var(
        "resnetv24_stage3_conv18_weight", shape=(1024, 256, 1, 1), dtype="float32")
    resnetv24_stage4_batchnorm0_gamma = relay.var(
        "resnetv24_stage4_batchnorm0_gamma", shape=(1024, ), dtype="float32")
    resnetv24_stage4_batchnorm0_beta = relay.var(
        "resnetv24_stage4_batchnorm0_beta", shape=(1024, ), dtype="float32")
    resnetv24_stage4_batchnorm0_running_mean = relay.var(
        "resnetv24_stage4_batchnorm0_running_mean", shape=(1024, ), dtype="float32")
    resnetv24_stage4_batchnorm0_running_var = relay.var(
        "resnetv24_stage4_batchnorm0_running_var", shape=(1024, ), dtype="float32")
    resnetv24_stage4_conv0_weight = relay.var(
        "resnetv24_stage4_conv0_weight", shape=(512, 1024, 1, 1), dtype="float32")
    resnetv24_stage4_batchnorm1_gamma = relay.var(
        "resnetv24_stage4_batchnorm1_gamma", shape=(512, ), dtype="float32")
    resnetv24_stage4_batchnorm1_beta = relay.var(
        "resnetv24_stage4_batchnorm1_beta", shape=(512, ), dtype="float32")
    resnetv24_stage4_batchnorm1_running_mean = relay.var(
        "resnetv24_stage4_batchnorm1_running_mean", shape=(512, ), dtype="float32")
    resnetv24_stage4_batchnorm1_running_var = relay.var(
        "resnetv24_stage4_batchnorm1_running_var", shape=(512, ), dtype="float32")
    resnetv24_stage4_conv1_weight = relay.var(
        "resnetv24_stage4_conv1_weight", shape=(512, 512, 3, 3), dtype="float32")
    resnetv24_stage4_batchnorm2_gamma = relay.var(
        "resnetv24_stage4_batchnorm2_gamma", shape=(512, ), dtype="float32")
    resnetv24_stage4_batchnorm2_beta = relay.var(
        "resnetv24_stage4_batchnorm2_beta", shape=(512, ), dtype="float32")
    resnetv24_stage4_batchnorm2_running_mean = relay.var(
        "resnetv24_stage4_batchnorm2_running_mean", shape=(512, ), dtype="float32")
    resnetv24_stage4_batchnorm2_running_var = relay.var(
        "resnetv24_stage4_batchnorm2_running_var", shape=(512, ), dtype="float32")
    resnetv24_stage4_conv2_weight = relay.var(
        "resnetv24_stage4_conv2_weight", shape=(2048, 512, 1, 1), dtype="float32")
    resnetv24_stage4_conv3_weight = relay.var(
        "resnetv24_stage4_conv3_weight", shape=(2048, 1024, 1, 1), dtype="float32")
    resnetv24_stage4_batchnorm3_gamma = relay.var(
        "resnetv24_stage4_batchnorm3_gamma", shape=(2048, ), dtype="float32")
    resnetv24_stage4_batchnorm3_beta = relay.var(
        "resnetv24_stage4_batchnorm3_beta", shape=(2048, ), dtype="float32")
    resnetv24_stage4_batchnorm3_running_mean = relay.var(
        "resnetv24_stage4_batchnorm3_running_mean", shape=(2048, ), dtype="float32")
    resnetv24_stage4_batchnorm3_running_var = relay.var(
        "resnetv24_stage4_batchnorm3_running_var", shape=(2048, ), dtype="float32")
    resnetv24_stage4_conv4_weight = relay.var(
        "resnetv24_stage4_conv4_weight", shape=(512, 2048, 1, 1), dtype="float32")
    resnetv24_stage4_batchnorm4_gamma = relay.var(
        "resnetv24_stage4_batchnorm4_gamma", shape=(512, ), dtype="float32")
    resnetv24_stage4_batchnorm4_beta = relay.var(
        "resnetv24_stage4_batchnorm4_beta", shape=(512, ), dtype="float32")
    resnetv24_stage4_batchnorm4_running_mean = relay.var(
        "resnetv24_stage4_batchnorm4_running_mean", shape=(512, ), dtype="float32")
    resnetv24_stage4_batchnorm4_running_var = relay.var(
        "resnetv24_stage4_batchnorm4_running_var", shape=(512, ), dtype="float32")
    resnetv24_stage4_conv5_weight = relay.var(
        "resnetv24_stage4_conv5_weight", shape=(512, 512, 3, 3), dtype="float32")
    resnetv24_stage4_batchnorm5_gamma = relay.var(
        "resnetv24_stage4_batchnorm5_gamma", shape=(512, ), dtype="float32")
    resnetv24_stage4_batchnorm5_beta = relay.var(
        "resnetv24_stage4_batchnorm5_beta", shape=(512, ), dtype="float32")
    resnetv24_stage4_batchnorm5_running_mean = relay.var(
        "resnetv24_stage4_batchnorm5_running_mean", shape=(512, ), dtype="float32")
    resnetv24_stage4_batchnorm5_running_var = relay.var(
        "resnetv24_stage4_batchnorm5_running_var", shape=(512, ), dtype="float32")
    resnetv24_stage4_conv6_weight = relay.var(
        "resnetv24_stage4_conv6_weight", shape=(2048, 512, 1, 1), dtype="float32")
    resnetv24_stage4_batchnorm6_gamma = relay.var(
        "resnetv24_stage4_batchnorm6_gamma", shape=(2048, ), dtype="float32")
    resnetv24_stage4_batchnorm6_beta = relay.var(
        "resnetv24_stage4_batchnorm6_beta", shape=(2048, ), dtype="float32")
    resnetv24_stage4_batchnorm6_running_mean = relay.var(
        "resnetv24_stage4_batchnorm6_running_mean", shape=(2048, ), dtype="float32")
    resnetv24_stage4_batchnorm6_running_var = relay.var(
        "resnetv24_stage4_batchnorm6_running_var", shape=(2048, ), dtype="float32")
    resnetv24_stage4_conv7_weight = relay.var(
        "resnetv24_stage4_conv7_weight", shape=(512, 2048, 1, 1), dtype="float32")
    resnetv24_stage4_batchnorm7_gamma = relay.var(
        "resnetv24_stage4_batchnorm7_gamma", shape=(512, ), dtype="float32")
    resnetv24_stage4_batchnorm7_beta = relay.var(
        "resnetv24_stage4_batchnorm7_beta", shape=(512, ), dtype="float32")
    resnetv24_stage4_batchnorm7_running_mean = relay.var(
        "resnetv24_stage4_batchnorm7_running_mean", shape=(512, ), dtype="float32")
    resnetv24_stage4_batchnorm7_running_var = relay.var(
        "resnetv24_stage4_batchnorm7_running_var", shape=(512, ), dtype="float32")
    resnetv24_stage4_conv8_weight = relay.var(
        "resnetv24_stage4_conv8_weight", shape=(512, 512, 3, 3), dtype="float32")
    resnetv24_stage4_batchnorm8_gamma = relay.var(
        "resnetv24_stage4_batchnorm8_gamma", shape=(512, ), dtype="float32")
    resnetv24_stage4_batchnorm8_beta = relay.var(
        "resnetv24_stage4_batchnorm8_beta", shape=(512, ), dtype="float32")
    resnetv24_stage4_batchnorm8_running_mean = relay.var(
        "resnetv24_stage4_batchnorm8_running_mean", shape=(512, ), dtype="float32")
    resnetv24_stage4_batchnorm8_running_var = relay.var(
        "resnetv24_stage4_batchnorm8_running_var", shape=(512, ), dtype="float32")
    resnetv24_stage4_conv9_weight = relay.var(
        "resnetv24_stage4_conv9_weight", shape=(2048, 512, 1, 1), dtype="float32")
    resnetv24_batchnorm2_gamma = relay.var(
        "resnetv24_batchnorm2_gamma", shape=(2048, ), dtype="float32")
    resnetv24_batchnorm2_beta = relay.var(
        "resnetv24_batchnorm2_beta", shape=(2048, ), dtype="float32")
    resnetv24_batchnorm2_running_mean = relay.var(
        "resnetv24_batchnorm2_running_mean", shape=(2048, ), dtype="float32")
    resnetv24_batchnorm2_running_var = relay.var(
        "resnetv24_batchnorm2_running_var", shape=(2048, ), dtype="float32")
    reshape_attr_tensor430 = relay.var(
        "reshape_attr_tensor430", shape=(2, ), dtype="float32")
    resnetv24_dense0_weight = relay.var(
        "resnetv24_dense0_weight", shape=(1000, 2048), dtype="float32")
    resnetv24_dense0_bias = relay.var(
        "resnetv24_dense0_bias", shape=(1000, ), dtype="float32")

    call_0_0 = relay.nn.batch_norm(data, resnetv24_batchnorm0_gamma, resnetv24_batchnorm0_beta,
                                   resnetv24_batchnorm0_running_mean, resnetv24_batchnorm0_running_var)
    call_2 = relay.nn.conv2d(call_0_0[0], resnetv24_conv0_weight, strides=[
                             2, 2], padding=[3, 3, 3, 3], channels=64, kernel_size=[7, 7])
    call_3_0 = relay.nn.batch_norm(call_2, resnetv24_batchnorm1_gamma, resnetv24_batchnorm1_beta,
                                   resnetv24_batchnorm1_running_mean, resnetv24_batchnorm1_running_var)
    call_5 = relay.nn.relu(call_3_0[0])
    call_6 = relay.nn.max_pool2d(call_5, pool_size=[3, 3], strides=[
                                 2, 2], padding=[1, 1, 1, 1])
    call_7_0 = relay.nn.batch_norm(call_6, resnetv24_stage1_batchnorm0_gamma, resnetv24_stage1_batchnorm0_beta,
                                   resnetv24_stage1_batchnorm0_running_mean, resnetv24_stage1_batchnorm0_running_var)
    call_9 = relay.nn.relu(call_7_0[0])
    call_10 = relay.nn.conv2d(call_9, resnetv24_stage1_conv0_weight, padding=[
                              0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_11_0 = relay.nn.batch_norm(call_10, resnetv24_stage1_batchnorm1_gamma, resnetv24_stage1_batchnorm1_beta,
                                    resnetv24_stage1_batchnorm1_running_mean, resnetv24_stage1_batchnorm1_running_var)
    call_13 = relay.nn.relu(call_11_0[0])
    call_14 = relay.nn.conv2d(call_13, resnetv24_stage1_conv1_weight, padding=[
                              1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_15_0 = relay.nn.batch_norm(call_14, resnetv24_stage1_batchnorm2_gamma, resnetv24_stage1_batchnorm2_beta,
                                    resnetv24_stage1_batchnorm2_running_mean, resnetv24_stage1_batchnorm2_running_var)
    call_17 = relay.nn.relu(call_15_0[0])
    call_18 = relay.nn.conv2d(call_17, resnetv24_stage1_conv2_weight, padding=[
                              0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_19 = relay.nn.conv2d(call_9, resnetv24_stage1_conv3_weight, padding=[
                              0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_20 = relay.add(call_18, call_19)
    call_21_0 = relay.nn.batch_norm(call_20, resnetv24_stage1_batchnorm3_gamma, resnetv24_stage1_batchnorm3_beta,
                                    resnetv24_stage1_batchnorm3_running_mean, resnetv24_stage1_batchnorm3_running_var)
    call_23 = relay.nn.relu(call_21_0[0])
    call_24 = relay.nn.conv2d(call_23, resnetv24_stage1_conv4_weight, padding=[
                              0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_25_0 = relay.nn.batch_norm(call_24, resnetv24_stage1_batchnorm4_gamma, resnetv24_stage1_batchnorm4_beta,
                                    resnetv24_stage1_batchnorm4_running_mean, resnetv24_stage1_batchnorm4_running_var)
    call_27 = relay.nn.relu(call_25_0[0])
    call_28 = relay.nn.conv2d(call_27, resnetv24_stage1_conv5_weight, padding=[
                              1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_29_0 = relay.nn.batch_norm(call_28, resnetv24_stage1_batchnorm5_gamma, resnetv24_stage1_batchnorm5_beta,
                                    resnetv24_stage1_batchnorm5_running_mean, resnetv24_stage1_batchnorm5_running_var)
    call_31 = relay.nn.relu(call_29_0[0])
    call_32 = relay.nn.conv2d(call_31, resnetv24_stage1_conv6_weight, padding=[
                              0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_33 = relay.add(call_32, call_20)
    call_34_0 = relay.nn.batch_norm(call_33, resnetv24_stage1_batchnorm6_gamma, resnetv24_stage1_batchnorm6_beta,
                                    resnetv24_stage1_batchnorm6_running_mean, resnetv24_stage1_batchnorm6_running_var)
    call_36 = relay.nn.relu(call_34_0[0])
    call_37 = relay.nn.conv2d(call_36, resnetv24_stage1_conv7_weight, padding=[
                              0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_38_0 = relay.nn.batch_norm(call_37, resnetv24_stage1_batchnorm7_gamma, resnetv24_stage1_batchnorm7_beta,
                                    resnetv24_stage1_batchnorm7_running_mean, resnetv24_stage1_batchnorm7_running_var)
    call_40 = relay.nn.relu(call_38_0[0])
    call_41 = relay.nn.conv2d(call_40, resnetv24_stage1_conv8_weight, padding=[
                              1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_42_0 = relay.nn.batch_norm(call_41, resnetv24_stage1_batchnorm8_gamma, resnetv24_stage1_batchnorm8_beta,
                                    resnetv24_stage1_batchnorm8_running_mean, resnetv24_stage1_batchnorm8_running_var)
    call_44 = relay.nn.relu(call_42_0[0])
    call_45 = relay.nn.conv2d(call_44, resnetv24_stage1_conv9_weight, padding=[
                              0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_46 = relay.add(call_45, call_33)
    call_47_0 = relay.nn.batch_norm(call_46, resnetv24_stage2_batchnorm0_gamma, resnetv24_stage2_batchnorm0_beta,
                                    resnetv24_stage2_batchnorm0_running_mean, resnetv24_stage2_batchnorm0_running_var)
    call_49 = relay.nn.relu(call_47_0[0])
    call_50 = relay.nn.conv2d(call_49, resnetv24_stage2_conv0_weight, padding=[
                              0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_51_0 = relay.nn.batch_norm(call_50, resnetv24_stage2_batchnorm1_gamma, resnetv24_stage2_batchnorm1_beta,
                                    resnetv24_stage2_batchnorm1_running_mean, resnetv24_stage2_batchnorm1_running_var)
    call_53 = relay.nn.relu(call_51_0[0])
    call_54 = relay.nn.conv2d(call_53, resnetv24_stage2_conv1_weight, strides=[
                              2, 2], padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_55_0 = relay.nn.batch_norm(call_54, resnetv24_stage2_batchnorm2_gamma, resnetv24_stage2_batchnorm2_beta,
                                    resnetv24_stage2_batchnorm2_running_mean, resnetv24_stage2_batchnorm2_running_var)
    call_57 = relay.nn.relu(call_55_0[0])
    call_58 = relay.nn.conv2d(call_57, resnetv24_stage2_conv2_weight, padding=[
                              0, 0, 0, 0], channels=512, kernel_size=[1, 1])
    call_59 = relay.nn.conv2d(call_49, resnetv24_stage2_conv3_weight, strides=[
                              2, 2], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1])
    call_60 = relay.add(call_58, call_59)
    call_61_0 = relay.nn.batch_norm(call_60, resnetv24_stage2_batchnorm3_gamma, resnetv24_stage2_batchnorm3_beta,
                                    resnetv24_stage2_batchnorm3_running_mean, resnetv24_stage2_batchnorm3_running_var)
    call_63 = relay.nn.relu(call_61_0[0])
    call_64 = relay.nn.conv2d(call_63, resnetv24_stage2_conv4_weight, padding=[
                              0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_65_0 = relay.nn.batch_norm(call_64, resnetv24_stage2_batchnorm4_gamma, resnetv24_stage2_batchnorm4_beta,
                                    resnetv24_stage2_batchnorm4_running_mean, resnetv24_stage2_batchnorm4_running_var)
    call_67 = relay.nn.relu(call_65_0[0])
    call_68 = relay.nn.conv2d(call_67, resnetv24_stage2_conv5_weight, padding=[
                              1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_69_0 = relay.nn.batch_norm(call_68, resnetv24_stage2_batchnorm5_gamma, resnetv24_stage2_batchnorm5_beta,
                                    resnetv24_stage2_batchnorm5_running_mean, resnetv24_stage2_batchnorm5_running_var)
    call_71 = relay.nn.relu(call_69_0[0])
    call_72 = relay.nn.conv2d(call_71, resnetv24_stage2_conv6_weight, padding=[
                              0, 0, 0, 0], channels=512, kernel_size=[1, 1])
    call_73 = relay.add(call_72, call_60)
    call_74_0 = relay.nn.batch_norm(call_73, resnetv24_stage2_batchnorm6_gamma, resnetv24_stage2_batchnorm6_beta,
                                    resnetv24_stage2_batchnorm6_running_mean, resnetv24_stage2_batchnorm6_running_var)
    call_76 = relay.nn.relu(call_74_0[0])
    call_77 = relay.nn.conv2d(call_76, resnetv24_stage2_conv7_weight, padding=[
                              0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_78_0 = relay.nn.batch_norm(call_77, resnetv24_stage2_batchnorm7_gamma, resnetv24_stage2_batchnorm7_beta,
                                    resnetv24_stage2_batchnorm7_running_mean, resnetv24_stage2_batchnorm7_running_var)
    call_80 = relay.nn.relu(call_78_0[0])
    call_81 = relay.nn.conv2d(call_80, resnetv24_stage2_conv8_weight, padding=[
                              1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_82_0 = relay.nn.batch_norm(call_81, resnetv24_stage2_batchnorm8_gamma, resnetv24_stage2_batchnorm8_beta,
                                    resnetv24_stage2_batchnorm8_running_mean, resnetv24_stage2_batchnorm8_running_var)
    call_84 = relay.nn.relu(call_82_0[0])
    call_85 = relay.nn.conv2d(call_84, resnetv24_stage2_conv9_weight, padding=[
                              0, 0, 0, 0], channels=512, kernel_size=[1, 1])
    call_86 = relay.add(call_85, call_73)
    call_87_0 = relay.nn.batch_norm(call_86, resnetv24_stage2_batchnorm9_gamma, resnetv24_stage2_batchnorm9_beta,
                                    resnetv24_stage2_batchnorm9_running_mean, resnetv24_stage2_batchnorm9_running_var)
    call_89 = relay.nn.relu(call_87_0[0])
    call_90 = relay.nn.conv2d(call_89, resnetv24_stage2_conv10_weight, padding=[
                              0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_91_0 = relay.nn.batch_norm(call_90, resnetv24_stage2_batchnorm10_gamma, resnetv24_stage2_batchnorm10_beta,
                                    resnetv24_stage2_batchnorm10_running_mean, resnetv24_stage2_batchnorm10_running_var)
    call_93 = relay.nn.relu(call_91_0[0])
    call_94 = relay.nn.conv2d(call_93, resnetv24_stage2_conv11_weight, padding=[
                              1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_95_0 = relay.nn.batch_norm(call_94, resnetv24_stage2_batchnorm11_gamma, resnetv24_stage2_batchnorm11_beta,
                                    resnetv24_stage2_batchnorm11_running_mean, resnetv24_stage2_batchnorm11_running_var)
    call_97 = relay.nn.relu(call_95_0[0])
    call_98 = relay.nn.conv2d(call_97, resnetv24_stage2_conv12_weight, padding=[
                              0, 0, 0, 0], channels=512, kernel_size=[1, 1])
    call_99 = relay.add(call_98, call_86)
    call_100_0 = relay.nn.batch_norm(call_99, resnetv24_stage3_batchnorm0_gamma, resnetv24_stage3_batchnorm0_beta,
                                     resnetv24_stage3_batchnorm0_running_mean, resnetv24_stage3_batchnorm0_running_var)
    call_102 = relay.nn.relu(call_100_0[0])
    call_103 = relay.nn.conv2d(call_102, resnetv24_stage3_conv0_weight, padding=[
                               0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_104_0 = relay.nn.batch_norm(call_103, resnetv24_stage3_batchnorm1_gamma, resnetv24_stage3_batchnorm1_beta,
                                     resnetv24_stage3_batchnorm1_running_mean, resnetv24_stage3_batchnorm1_running_var)
    call_106 = relay.nn.relu(call_104_0[0])
    call_107 = relay.nn.conv2d(call_106, resnetv24_stage3_conv1_weight, strides=[
                               2, 2], padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_108_0 = relay.nn.batch_norm(call_107, resnetv24_stage3_batchnorm2_gamma, resnetv24_stage3_batchnorm2_beta,
                                     resnetv24_stage3_batchnorm2_running_mean, resnetv24_stage3_batchnorm2_running_var)
    call_110 = relay.nn.relu(call_108_0[0])
    call_111 = relay.nn.conv2d(call_110, resnetv24_stage3_conv2_weight, padding=[
                               0, 0, 0, 0], channels=1024, kernel_size=[1, 1])
    call_112 = relay.nn.conv2d(call_102, resnetv24_stage3_conv3_weight, strides=[
                               2, 2], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1])
    call_113 = relay.add(call_111, call_112)
    call_114_0 = relay.nn.batch_norm(call_113, resnetv24_stage3_batchnorm3_gamma, resnetv24_stage3_batchnorm3_beta,
                                     resnetv24_stage3_batchnorm3_running_mean, resnetv24_stage3_batchnorm3_running_var)
    call_116 = relay.nn.relu(call_114_0[0])
    call_117 = relay.nn.conv2d(call_116, resnetv24_stage3_conv4_weight, padding=[
                               0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_118_0 = relay.nn.batch_norm(call_117, resnetv24_stage3_batchnorm4_gamma, resnetv24_stage3_batchnorm4_beta,
                                     resnetv24_stage3_batchnorm4_running_mean, resnetv24_stage3_batchnorm4_running_var)
    call_120 = relay.nn.relu(call_118_0[0])
    call_121 = relay.nn.conv2d(call_120, resnetv24_stage3_conv5_weight, padding=[
                               1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_122_0 = relay.nn.batch_norm(call_121, resnetv24_stage3_batchnorm5_gamma, resnetv24_stage3_batchnorm5_beta,
                                     resnetv24_stage3_batchnorm5_running_mean, resnetv24_stage3_batchnorm5_running_var)
    call_124 = relay.nn.relu(call_122_0[0])
    call_125 = relay.nn.conv2d(call_124, resnetv24_stage3_conv6_weight, padding=[
                               0, 0, 0, 0], channels=1024, kernel_size=[1, 1])
    call_126 = relay.add(call_125, call_113)
    call_127_0 = relay.nn.batch_norm(call_126, resnetv24_stage3_batchnorm6_gamma, resnetv24_stage3_batchnorm6_beta,
                                     resnetv24_stage3_batchnorm6_running_mean, resnetv24_stage3_batchnorm6_running_var)
    call_129 = relay.nn.relu(call_127_0[0])
    call_130 = relay.nn.conv2d(call_129, resnetv24_stage3_conv7_weight, padding=[
                               0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_131_0 = relay.nn.batch_norm(call_130, resnetv24_stage3_batchnorm7_gamma, resnetv24_stage3_batchnorm7_beta,
                                     resnetv24_stage3_batchnorm7_running_mean, resnetv24_stage3_batchnorm7_running_var)
    call_133 = relay.nn.relu(call_131_0[0])
    call_134 = relay.nn.conv2d(call_133, resnetv24_stage3_conv8_weight, padding=[
                               1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_135_0 = relay.nn.batch_norm(call_134, resnetv24_stage3_batchnorm8_gamma, resnetv24_stage3_batchnorm8_beta,
                                     resnetv24_stage3_batchnorm8_running_mean, resnetv24_stage3_batchnorm8_running_var)
    call_137 = relay.nn.relu(call_135_0[0])
    call_138 = relay.nn.conv2d(call_137, resnetv24_stage3_conv9_weight, padding=[
                               0, 0, 0, 0], channels=1024, kernel_size=[1, 1])
    call_139 = relay.add(call_138, call_126)
    call_140_0 = relay.nn.batch_norm(call_139, resnetv24_stage3_batchnorm9_gamma, resnetv24_stage3_batchnorm9_beta,
                                     resnetv24_stage3_batchnorm9_running_mean, resnetv24_stage3_batchnorm9_running_var)
    call_142 = relay.nn.relu(call_140_0[0])
    call_143 = relay.nn.conv2d(call_142, resnetv24_stage3_conv10_weight, padding=[
                               0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_144_0 = relay.nn.batch_norm(call_143, resnetv24_stage3_batchnorm10_gamma, resnetv24_stage3_batchnorm10_beta,
                                     resnetv24_stage3_batchnorm10_running_mean, resnetv24_stage3_batchnorm10_running_var)
    call_146 = relay.nn.relu(call_144_0[0])
    call_147 = relay.nn.conv2d(call_146, resnetv24_stage3_conv11_weight, padding=[
                               1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_148_0 = relay.nn.batch_norm(call_147, resnetv24_stage3_batchnorm11_gamma, resnetv24_stage3_batchnorm11_beta,
                                     resnetv24_stage3_batchnorm11_running_mean, resnetv24_stage3_batchnorm11_running_var)
    call_150 = relay.nn.relu(call_148_0[0])
    call_151 = relay.nn.conv2d(call_150, resnetv24_stage3_conv12_weight, padding=[
                               0, 0, 0, 0], channels=1024, kernel_size=[1, 1])
    call_152 = relay.add(call_151, call_139)
    call_153_0 = relay.nn.batch_norm(call_152, resnetv24_stage3_batchnorm12_gamma, resnetv24_stage3_batchnorm12_beta,
                                     resnetv24_stage3_batchnorm12_running_mean, resnetv24_stage3_batchnorm12_running_var)
    call_155 = relay.nn.relu(call_153_0[0])
    call_156 = relay.nn.conv2d(call_155, resnetv24_stage3_conv13_weight, padding=[
                               0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_157_0 = relay.nn.batch_norm(call_156, resnetv24_stage3_batchnorm13_gamma, resnetv24_stage3_batchnorm13_beta,
                                     resnetv24_stage3_batchnorm13_running_mean, resnetv24_stage3_batchnorm13_running_var)
    call_159 = relay.nn.relu(call_157_0[0])
    call_160 = relay.nn.conv2d(call_159, resnetv24_stage3_conv14_weight, padding=[
                               1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_161_0 = relay.nn.batch_norm(call_160, resnetv24_stage3_batchnorm14_gamma, resnetv24_stage3_batchnorm14_beta,
                                     resnetv24_stage3_batchnorm14_running_mean, resnetv24_stage3_batchnorm14_running_var)
    call_163 = relay.nn.relu(call_161_0[0])
    call_164 = relay.nn.conv2d(call_163, resnetv24_stage3_conv15_weight, padding=[
                               0, 0, 0, 0], channels=1024, kernel_size=[1, 1])
    call_165 = relay.add(call_164, call_152)
    call_166_0 = relay.nn.batch_norm(call_165, resnetv24_stage3_batchnorm15_gamma, resnetv24_stage3_batchnorm15_beta,
                                     resnetv24_stage3_batchnorm15_running_mean, resnetv24_stage3_batchnorm15_running_var)
    call_168 = relay.nn.relu(call_166_0[0])
    call_169 = relay.nn.conv2d(call_168, resnetv24_stage3_conv16_weight, padding=[
                               0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_170_0 = relay.nn.batch_norm(call_169, resnetv24_stage3_batchnorm16_gamma, resnetv24_stage3_batchnorm16_beta,
                                     resnetv24_stage3_batchnorm16_running_mean, resnetv24_stage3_batchnorm16_running_var)
    call_172 = relay.nn.relu(call_170_0[0])
    call_173 = relay.nn.conv2d(call_172, resnetv24_stage3_conv17_weight, padding=[
                               1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_174_0 = relay.nn.batch_norm(call_173, resnetv24_stage3_batchnorm17_gamma, resnetv24_stage3_batchnorm17_beta,
                                     resnetv24_stage3_batchnorm17_running_mean, resnetv24_stage3_batchnorm17_running_var)
    call_176 = relay.nn.relu(call_174_0[0])
    call_177 = relay.nn.conv2d(call_176, resnetv24_stage3_conv18_weight, padding=[
                               0, 0, 0, 0], channels=1024, kernel_size=[1, 1])
    call_178 = relay.add(call_177, call_165)
    call_179_0 = relay.nn.batch_norm(call_178, resnetv24_stage4_batchnorm0_gamma, resnetv24_stage4_batchnorm0_beta,
                                     resnetv24_stage4_batchnorm0_running_mean, resnetv24_stage4_batchnorm0_running_var)
    call_181 = relay.nn.relu(call_179_0[0])
    call_182 = relay.nn.conv2d(call_181, resnetv24_stage4_conv0_weight, padding=[
                               0, 0, 0, 0], channels=512, kernel_size=[1, 1])
    call_183_0 = relay.nn.batch_norm(call_182, resnetv24_stage4_batchnorm1_gamma, resnetv24_stage4_batchnorm1_beta,
                                     resnetv24_stage4_batchnorm1_running_mean, resnetv24_stage4_batchnorm1_running_var)
    call_185 = relay.nn.relu(call_183_0[0])
    call_186 = relay.nn.conv2d(call_185, resnetv24_stage4_conv1_weight, strides=[
                               2, 2], padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_187_0 = relay.nn.batch_norm(call_186, resnetv24_stage4_batchnorm2_gamma, resnetv24_stage4_batchnorm2_beta,
                                     resnetv24_stage4_batchnorm2_running_mean, resnetv24_stage4_batchnorm2_running_var)
    call_189 = relay.nn.relu(call_187_0[0])
    call_190 = relay.nn.conv2d(call_189, resnetv24_stage4_conv2_weight, padding=[
                               0, 0, 0, 0], channels=2048, kernel_size=[1, 1])
    call_191 = relay.nn.conv2d(call_181, resnetv24_stage4_conv3_weight, strides=[
                               2, 2], padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1])
    call_192 = relay.add(call_190, call_191)
    call_193_0 = relay.nn.batch_norm(call_192, resnetv24_stage4_batchnorm3_gamma, resnetv24_stage4_batchnorm3_beta,
                                     resnetv24_stage4_batchnorm3_running_mean, resnetv24_stage4_batchnorm3_running_var)
    call_195 = relay.nn.relu(call_193_0[0])
    call_196 = relay.nn.conv2d(call_195, resnetv24_stage4_conv4_weight, padding=[
                               0, 0, 0, 0], channels=512, kernel_size=[1, 1])
    call_197_0 = relay.nn.batch_norm(call_196, resnetv24_stage4_batchnorm4_gamma, resnetv24_stage4_batchnorm4_beta,
                                     resnetv24_stage4_batchnorm4_running_mean, resnetv24_stage4_batchnorm4_running_var)
    call_199 = relay.nn.relu(call_197_0[0])
    call_200 = relay.nn.conv2d(call_199, resnetv24_stage4_conv5_weight, padding=[
                               1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_201_0 = relay.nn.batch_norm(call_200, resnetv24_stage4_batchnorm5_gamma, resnetv24_stage4_batchnorm5_beta,
                                     resnetv24_stage4_batchnorm5_running_mean, resnetv24_stage4_batchnorm5_running_var)
    call_203 = relay.nn.relu(call_201_0[0])
    call_204 = relay.nn.conv2d(call_203, resnetv24_stage4_conv6_weight, padding=[
                               0, 0, 0, 0], channels=2048, kernel_size=[1, 1])
    call_205 = relay.add(call_204, call_192)
    call_206_0 = relay.nn.batch_norm(call_205, resnetv24_stage4_batchnorm6_gamma, resnetv24_stage4_batchnorm6_beta,
                                     resnetv24_stage4_batchnorm6_running_mean, resnetv24_stage4_batchnorm6_running_var)
    call_208 = relay.nn.relu(call_206_0[0])
    call_209 = relay.nn.conv2d(call_208, resnetv24_stage4_conv7_weight, padding=[
                               0, 0, 0, 0], channels=512, kernel_size=[1, 1])
    call_210_0 = relay.nn.batch_norm(call_209, resnetv24_stage4_batchnorm7_gamma, resnetv24_stage4_batchnorm7_beta,
                                     resnetv24_stage4_batchnorm7_running_mean, resnetv24_stage4_batchnorm7_running_var)
    call_212 = relay.nn.relu(call_210_0[0])
    call_213 = relay.nn.conv2d(call_212, resnetv24_stage4_conv8_weight, padding=[
                               1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_214_0 = relay.nn.batch_norm(call_213, resnetv24_stage4_batchnorm8_gamma, resnetv24_stage4_batchnorm8_beta,
                                     resnetv24_stage4_batchnorm8_running_mean, resnetv24_stage4_batchnorm8_running_var)
    call_216 = relay.nn.relu(call_214_0[0])
    call_217 = relay.nn.conv2d(call_216, resnetv24_stage4_conv9_weight, padding=[
                               0, 0, 0, 0], channels=2048, kernel_size=[1, 1])
    call_218 = relay.add(call_217, call_205)
    call_219_0 = relay.nn.batch_norm(call_218, resnetv24_batchnorm2_gamma, resnetv24_batchnorm2_beta,
                                     resnetv24_batchnorm2_running_mean, resnetv24_batchnorm2_running_var)
    call_221 = relay.nn.relu(call_219_0[0])
    call_222 = relay.nn.global_avg_pool2d(call_221)
    call_223 = relay.reshape(call_222, newshape=[0, -1])
    call_224 = relay.nn.batch_flatten(call_223)
    call_225 = relay.nn.dense(call_224, resnetv24_dense0_weight, units=1000)
    call_226 = relay.multiply(relay.const(1, "float32"), resnetv24_dense0_bias)
    call_output0 = relay.add(call_225, call_226)
    return call_output0
