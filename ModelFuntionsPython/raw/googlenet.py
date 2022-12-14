# this file is create by program.def GoogleNet(pre_input=None):    import tvm    import numpy as np    from tvm import relay    data_0 = pre_input if pre_input is not None else relay.var("data_0", shape=(15, 3, 224, 224), dtype="float32")    fc_weight = relay.var("fc.weight", shape=(1000, 1024), dtype="float32")    fc_bias = relay.var("fc.bias", shape=(1000, ), dtype="float32")    onnx_Conv_567 = relay.var("onnx_Conv_567", shape=(64, 3, 7, 7), dtype="float32")    onnx_Conv_568 = relay.var("onnx_Conv_568", shape=(64, ), dtype="float32")    onnx_Conv_570 = relay.var("onnx_Conv_570", shape=(64, 64, 1, 1), dtype="float32")    onnx_Conv_571 = relay.var("onnx_Conv_571", shape=(64, ), dtype="float32")    onnx_Conv_573 = relay.var("onnx_Conv_573", shape=(192, 64, 3, 3), dtype="float32")    onnx_Conv_574 = relay.var("onnx_Conv_574", shape=(192, ), dtype="float32")    onnx_Conv_576 = relay.var("onnx_Conv_576", shape=(64, 192, 1, 1), dtype="float32")    onnx_Conv_577 = relay.var("onnx_Conv_577", shape=(64, ), dtype="float32")    onnx_Conv_579 = relay.var("onnx_Conv_579", shape=(96, 192, 1, 1), dtype="float32")    onnx_Conv_580 = relay.var("onnx_Conv_580", shape=(96, ), dtype="float32")    onnx_Conv_582 = relay.var("onnx_Conv_582", shape=(128, 96, 3, 3), dtype="float32")    onnx_Conv_583 = relay.var("onnx_Conv_583", shape=(128, ), dtype="float32")    onnx_Conv_585 = relay.var("onnx_Conv_585", shape=(16, 192, 1, 1), dtype="float32")    onnx_Conv_586 = relay.var("onnx_Conv_586", shape=(16, ), dtype="float32")    onnx_Conv_588 = relay.var("onnx_Conv_588", shape=(32, 16, 3, 3), dtype="float32")    onnx_Conv_589 = relay.var("onnx_Conv_589", shape=(32, ), dtype="float32")    onnx_Conv_591 = relay.var("onnx_Conv_591", shape=(32, 192, 1, 1), dtype="float32")    onnx_Conv_592 = relay.var("onnx_Conv_592", shape=(32, ), dtype="float32")    onnx_Conv_594 = relay.var("onnx_Conv_594", shape=(128, 256, 1, 1), dtype="float32")    onnx_Conv_595 = relay.var("onnx_Conv_595", shape=(128, ), dtype="float32")    onnx_Conv_597 = relay.var("onnx_Conv_597", shape=(128, 256, 1, 1), dtype="float32")    onnx_Conv_598 = relay.var("onnx_Conv_598", shape=(128, ), dtype="float32")    onnx_Conv_600 = relay.var("onnx_Conv_600", shape=(192, 128, 3, 3), dtype="float32")    onnx_Conv_601 = relay.var("onnx_Conv_601", shape=(192, ), dtype="float32")    onnx_Conv_603 = relay.var("onnx_Conv_603", shape=(32, 256, 1, 1), dtype="float32")    onnx_Conv_604 = relay.var("onnx_Conv_604", shape=(32, ), dtype="float32")    onnx_Conv_606 = relay.var("onnx_Conv_606", shape=(96, 32, 3, 3), dtype="float32")    onnx_Conv_607 = relay.var("onnx_Conv_607", shape=(96, ), dtype="float32")    onnx_Conv_609 = relay.var("onnx_Conv_609", shape=(64, 256, 1, 1), dtype="float32")    onnx_Conv_610 = relay.var("onnx_Conv_610", shape=(64, ), dtype="float32")    onnx_Conv_612 = relay.var("onnx_Conv_612", shape=(192, 480, 1, 1), dtype="float32")    onnx_Conv_613 = relay.var("onnx_Conv_613", shape=(192, ), dtype="float32")    onnx_Conv_615 = relay.var("onnx_Conv_615", shape=(96, 480, 1, 1), dtype="float32")    onnx_Conv_616 = relay.var("onnx_Conv_616", shape=(96, ), dtype="float32")    onnx_Conv_618 = relay.var("onnx_Conv_618", shape=(208, 96, 3, 3), dtype="float32")    onnx_Conv_619 = relay.var("onnx_Conv_619", shape=(208, ), dtype="float32")    onnx_Conv_621 = relay.var("onnx_Conv_621", shape=(16, 480, 1, 1), dtype="float32")    onnx_Conv_622 = relay.var("onnx_Conv_622", shape=(16, ), dtype="float32")    onnx_Conv_624 = relay.var("onnx_Conv_624", shape=(48, 16, 3, 3), dtype="float32")    onnx_Conv_625 = relay.var("onnx_Conv_625", shape=(48, ), dtype="float32")    onnx_Conv_627 = relay.var("onnx_Conv_627", shape=(64, 480, 1, 1), dtype="float32")    onnx_Conv_628 = relay.var("onnx_Conv_628", shape=(64, ), dtype="float32")    onnx_Conv_630 = relay.var("onnx_Conv_630", shape=(160, 512, 1, 1), dtype="float32")    onnx_Conv_631 = relay.var("onnx_Conv_631", shape=(160, ), dtype="float32")    onnx_Conv_633 = relay.var("onnx_Conv_633", shape=(112, 512, 1, 1), dtype="float32")    onnx_Conv_634 = relay.var("onnx_Conv_634", shape=(112, ), dtype="float32")    onnx_Conv_636 = relay.var("onnx_Conv_636", shape=(224, 112, 3, 3), dtype="float32")    onnx_Conv_637 = relay.var("onnx_Conv_637", shape=(224, ), dtype="float32")    onnx_Conv_639 = relay.var("onnx_Conv_639", shape=(24, 512, 1, 1), dtype="float32")    onnx_Conv_640 = relay.var("onnx_Conv_640", shape=(24, ), dtype="float32")    onnx_Conv_642 = relay.var("onnx_Conv_642", shape=(64, 24, 3, 3), dtype="float32")    onnx_Conv_643 = relay.var("onnx_Conv_643", shape=(64, ), dtype="float32")    onnx_Conv_645 = relay.var("onnx_Conv_645", shape=(64, 512, 1, 1), dtype="float32")    onnx_Conv_646 = relay.var("onnx_Conv_646", shape=(64, ), dtype="float32")    onnx_Conv_648 = relay.var("onnx_Conv_648", shape=(128, 512, 1, 1), dtype="float32")    onnx_Conv_649 = relay.var("onnx_Conv_649", shape=(128, ), dtype="float32")    onnx_Conv_651 = relay.var("onnx_Conv_651", shape=(128, 512, 1, 1), dtype="float32")    onnx_Conv_652 = relay.var("onnx_Conv_652", shape=(128, ), dtype="float32")    onnx_Conv_654 = relay.var("onnx_Conv_654", shape=(256, 128, 3, 3), dtype="float32")    onnx_Conv_655 = relay.var("onnx_Conv_655", shape=(256, ), dtype="float32")    onnx_Conv_657 = relay.var("onnx_Conv_657", shape=(24, 512, 1, 1), dtype="float32")    onnx_Conv_658 = relay.var("onnx_Conv_658", shape=(24, ), dtype="float32")    onnx_Conv_660 = relay.var("onnx_Conv_660", shape=(64, 24, 3, 3), dtype="float32")    onnx_Conv_661 = relay.var("onnx_Conv_661", shape=(64, ), dtype="float32")    onnx_Conv_663 = relay.var("onnx_Conv_663", shape=(64, 512, 1, 1), dtype="float32")    onnx_Conv_664 = relay.var("onnx_Conv_664", shape=(64, ), dtype="float32")    onnx_Conv_666 = relay.var("onnx_Conv_666", shape=(112, 512, 1, 1), dtype="float32")    onnx_Conv_667 = relay.var("onnx_Conv_667", shape=(112, ), dtype="float32")    onnx_Conv_669 = relay.var("onnx_Conv_669", shape=(144, 512, 1, 1), dtype="float32")    onnx_Conv_670 = relay.var("onnx_Conv_670", shape=(144, ), dtype="float32")    onnx_Conv_672 = relay.var("onnx_Conv_672", shape=(288, 144, 3, 3), dtype="float32")    onnx_Conv_673 = relay.var("onnx_Conv_673", shape=(288, ), dtype="float32")    onnx_Conv_675 = relay.var("onnx_Conv_675", shape=(32, 512, 1, 1), dtype="float32")    onnx_Conv_676 = relay.var("onnx_Conv_676", shape=(32, ), dtype="float32")    onnx_Conv_678 = relay.var("onnx_Conv_678", shape=(64, 32, 3, 3), dtype="float32")    onnx_Conv_679 = relay.var("onnx_Conv_679", shape=(64, ), dtype="float32")    onnx_Conv_681 = relay.var("onnx_Conv_681", shape=(64, 512, 1, 1), dtype="float32")    onnx_Conv_682 = relay.var("onnx_Conv_682", shape=(64, ), dtype="float32")    onnx_Conv_684 = relay.var("onnx_Conv_684", shape=(256, 528, 1, 1), dtype="float32")    onnx_Conv_685 = relay.var("onnx_Conv_685", shape=(256, ), dtype="float32")    onnx_Conv_687 = relay.var("onnx_Conv_687", shape=(160, 528, 1, 1), dtype="float32")    onnx_Conv_688 = relay.var("onnx_Conv_688", shape=(160, ), dtype="float32")    onnx_Conv_690 = relay.var("onnx_Conv_690", shape=(320, 160, 3, 3), dtype="float32")    onnx_Conv_691 = relay.var("onnx_Conv_691", shape=(320, ), dtype="float32")    onnx_Conv_693 = relay.var("onnx_Conv_693", shape=(32, 528, 1, 1), dtype="float32")    onnx_Conv_694 = relay.var("onnx_Conv_694", shape=(32, ), dtype="float32")    onnx_Conv_696 = relay.var("onnx_Conv_696", shape=(128, 32, 3, 3), dtype="float32")    onnx_Conv_697 = relay.var("onnx_Conv_697", shape=(128, ), dtype="float32")    onnx_Conv_699 = relay.var("onnx_Conv_699", shape=(128, 528, 1, 1), dtype="float32")    onnx_Conv_700 = relay.var("onnx_Conv_700", shape=(128, ), dtype="float32")    onnx_Conv_702 = relay.var("onnx_Conv_702", shape=(256, 832, 1, 1), dtype="float32")    onnx_Conv_703 = relay.var("onnx_Conv_703", shape=(256, ), dtype="float32")    onnx_Conv_705 = relay.var("onnx_Conv_705", shape=(160, 832, 1, 1), dtype="float32")    onnx_Conv_706 = relay.var("onnx_Conv_706", shape=(160, ), dtype="float32")    onnx_Conv_708 = relay.var("onnx_Conv_708", shape=(320, 160, 3, 3), dtype="float32")    onnx_Conv_709 = relay.var("onnx_Conv_709", shape=(320, ), dtype="float32")    onnx_Conv_711 = relay.var("onnx_Conv_711", shape=(32, 832, 1, 1), dtype="float32")    onnx_Conv_712 = relay.var("onnx_Conv_712", shape=(32, ), dtype="float32")    onnx_Conv_714 = relay.var("onnx_Conv_714", shape=(128, 32, 3, 3), dtype="float32")    onnx_Conv_715 = relay.var("onnx_Conv_715", shape=(128, ), dtype="float32")    onnx_Conv_717 = relay.var("onnx_Conv_717", shape=(128, 832, 1, 1), dtype="float32")    onnx_Conv_718 = relay.var("onnx_Conv_718", shape=(128, ), dtype="float32")    onnx_Conv_720 = relay.var("onnx_Conv_720", shape=(384, 832, 1, 1), dtype="float32")    onnx_Conv_721 = relay.var("onnx_Conv_721", shape=(384, ), dtype="float32")    onnx_Conv_723 = relay.var("onnx_Conv_723", shape=(192, 832, 1, 1), dtype="float32")    onnx_Conv_724 = relay.var("onnx_Conv_724", shape=(192, ), dtype="float32")    onnx_Conv_726 = relay.var("onnx_Conv_726", shape=(384, 192, 3, 3), dtype="float32")    onnx_Conv_727 = relay.var("onnx_Conv_727", shape=(384, ), dtype="float32")    onnx_Conv_729 = relay.var("onnx_Conv_729", shape=(48, 832, 1, 1), dtype="float32")    onnx_Conv_730 = relay.var("onnx_Conv_730", shape=(48, ), dtype="float32")    onnx_Conv_732 = relay.var("onnx_Conv_732", shape=(128, 48, 3, 3), dtype="float32")    onnx_Conv_733 = relay.var("onnx_Conv_733", shape=(128, ), dtype="float32")    onnx_Conv_735 = relay.var("onnx_Conv_735", shape=(128, 832, 1, 1), dtype="float32")    onnx_Conv_736 = relay.var("onnx_Conv_736", shape=(128, ), dtype="float32")    call_0 = relay.take(data_0, relay.const(np.array(0, dtype="int64")), axis=1)
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
    call_13 = relay.concatenate(relay.Tuple([call_9, call_10, call_11]), axis=1)
    call_14 = relay.nn.conv2d(call_13, onnx_Conv_567, strides=[2, 2], padding=[3, 3, 3, 3], channels=64, kernel_size=[7, 7])
    call_15 = relay.nn.bias_add(call_14, onnx_Conv_568)
    call_16 = relay.nn.relu(call_15)
    call_17 = relay.nn.max_pool2d(call_16, pool_size=[3, 3], strides=[2, 2], padding=[0, 0, 0, 0], ceil_mode=True)
    call_18 = relay.nn.conv2d(call_17, onnx_Conv_570, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_19 = relay.nn.bias_add(call_18, onnx_Conv_571)
    call_20 = relay.nn.relu(call_19)
    call_21 = relay.nn.conv2d(call_20, onnx_Conv_573, padding=[1, 1, 1, 1], channels=192, kernel_size=[3, 3])
    call_22 = relay.nn.bias_add(call_21, onnx_Conv_574)
    call_23 = relay.nn.relu(call_22)
    call_24 = relay.nn.max_pool2d(call_23, pool_size=[3, 3], strides=[2, 2], padding=[0, 0, 0, 0], ceil_mode=True)
    call_25 = relay.nn.conv2d(call_24, onnx_Conv_576, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_26 = relay.nn.bias_add(call_25, onnx_Conv_577)
    call_27 = relay.nn.conv2d(call_24, onnx_Conv_579, padding=[0, 0, 0, 0], channels=96, kernel_size=[1, 1])
    call_28 = relay.nn.bias_add(call_27, onnx_Conv_580)
    call_29 = relay.nn.relu(call_28)
    call_30 = relay.nn.conv2d(call_29, onnx_Conv_582, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_31 = relay.nn.bias_add(call_30, onnx_Conv_583)
    call_32 = relay.nn.conv2d(call_24, onnx_Conv_585, padding=[0, 0, 0, 0], channels=16, kernel_size=[1, 1])
    call_33 = relay.nn.bias_add(call_32, onnx_Conv_586)
    call_34 = relay.nn.relu(call_33)
    call_35 = relay.nn.conv2d(call_34, onnx_Conv_588, padding=[1, 1, 1, 1], channels=32, kernel_size=[3, 3])
    call_36 = relay.nn.bias_add(call_35, onnx_Conv_589)
    call_37 = relay.nn.max_pool2d(call_24, pool_size=[3, 3], padding=[1, 1, 1, 1], ceil_mode=True)
    call_38 = relay.nn.conv2d(call_37, onnx_Conv_591, padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1])
    call_39 = relay.nn.bias_add(call_38, onnx_Conv_592)
    call_40 = relay.nn.relu(call_26)
    call_41 = relay.nn.relu(call_31)
    call_42 = relay.nn.relu(call_36)
    call_43 = relay.nn.relu(call_39)
    call_45 = relay.concatenate(relay.Tuple([call_40, call_41, call_42, call_43]), axis=1)
    call_46 = relay.nn.conv2d(call_45, onnx_Conv_594, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_47 = relay.nn.bias_add(call_46, onnx_Conv_595)
    call_48 = relay.nn.conv2d(call_45, onnx_Conv_597, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_49 = relay.nn.bias_add(call_48, onnx_Conv_598)
    call_50 = relay.nn.relu(call_49)
    call_51 = relay.nn.conv2d(call_50, onnx_Conv_600, padding=[1, 1, 1, 1], channels=192, kernel_size=[3, 3])
    call_52 = relay.nn.bias_add(call_51, onnx_Conv_601)
    call_53 = relay.nn.conv2d(call_45, onnx_Conv_603, padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1])
    call_54 = relay.nn.bias_add(call_53, onnx_Conv_604)
    call_55 = relay.nn.relu(call_54)
    call_56 = relay.nn.conv2d(call_55, onnx_Conv_606, padding=[1, 1, 1, 1], channels=96, kernel_size=[3, 3])
    call_57 = relay.nn.bias_add(call_56, onnx_Conv_607)
    call_58 = relay.nn.max_pool2d(call_45, pool_size=[3, 3], padding=[1, 1, 1, 1], ceil_mode=True)
    call_59 = relay.nn.conv2d(call_58, onnx_Conv_609, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_60 = relay.nn.bias_add(call_59, onnx_Conv_610)
    call_61 = relay.nn.relu(call_47)
    call_62 = relay.nn.relu(call_52)
    call_63 = relay.nn.relu(call_57)
    call_64 = relay.nn.relu(call_60)
    call_66 = relay.concatenate(relay.Tuple([call_61, call_62, call_63, call_64]), axis=1)
    call_67 = relay.nn.max_pool2d(call_66, pool_size=[3, 3], strides=[2, 2], padding=[0, 0, 0, 0], ceil_mode=True)
    call_68 = relay.nn.conv2d(call_67, onnx_Conv_612, padding=[0, 0, 0, 0], channels=192, kernel_size=[1, 1])
    call_69 = relay.nn.bias_add(call_68, onnx_Conv_613)
    call_70 = relay.nn.conv2d(call_67, onnx_Conv_615, padding=[0, 0, 0, 0], channels=96, kernel_size=[1, 1])
    call_71 = relay.nn.bias_add(call_70, onnx_Conv_616)
    call_72 = relay.nn.relu(call_71)
    call_73 = relay.nn.conv2d(call_72, onnx_Conv_618, padding=[1, 1, 1, 1], channels=208, kernel_size=[3, 3])
    call_74 = relay.nn.bias_add(call_73, onnx_Conv_619)
    call_75 = relay.nn.conv2d(call_67, onnx_Conv_621, padding=[0, 0, 0, 0], channels=16, kernel_size=[1, 1])
    call_76 = relay.nn.bias_add(call_75, onnx_Conv_622)
    call_77 = relay.nn.relu(call_76)
    call_78 = relay.nn.conv2d(call_77, onnx_Conv_624, padding=[1, 1, 1, 1], channels=48, kernel_size=[3, 3])
    call_79 = relay.nn.bias_add(call_78, onnx_Conv_625)
    call_80 = relay.nn.max_pool2d(call_67, pool_size=[3, 3], padding=[1, 1, 1, 1], ceil_mode=True)
    call_81 = relay.nn.conv2d(call_80, onnx_Conv_627, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_82 = relay.nn.bias_add(call_81, onnx_Conv_628)
    call_83 = relay.nn.relu(call_69)
    call_84 = relay.nn.relu(call_74)
    call_85 = relay.nn.relu(call_79)
    call_86 = relay.nn.relu(call_82)
    call_88 = relay.concatenate(relay.Tuple([call_83, call_84, call_85, call_86]), axis=1)
    call_89 = relay.nn.conv2d(call_88, onnx_Conv_630, padding=[0, 0, 0, 0], channels=160, kernel_size=[1, 1])
    call_90 = relay.nn.bias_add(call_89, onnx_Conv_631)
    call_91 = relay.nn.conv2d(call_88, onnx_Conv_633, padding=[0, 0, 0, 0], channels=112, kernel_size=[1, 1])
    call_92 = relay.nn.bias_add(call_91, onnx_Conv_634)
    call_93 = relay.nn.relu(call_92)
    call_94 = relay.nn.conv2d(call_93, onnx_Conv_636, padding=[1, 1, 1, 1], channels=224, kernel_size=[3, 3])
    call_95 = relay.nn.bias_add(call_94, onnx_Conv_637)
    call_96 = relay.nn.conv2d(call_88, onnx_Conv_639, padding=[0, 0, 0, 0], channels=24, kernel_size=[1, 1])
    call_97 = relay.nn.bias_add(call_96, onnx_Conv_640)
    call_98 = relay.nn.relu(call_97)
    call_99 = relay.nn.conv2d(call_98, onnx_Conv_642, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_100 = relay.nn.bias_add(call_99, onnx_Conv_643)
    call_101 = relay.nn.max_pool2d(call_88, pool_size=[3, 3], padding=[1, 1, 1, 1], ceil_mode=True)
    call_102 = relay.nn.conv2d(call_101, onnx_Conv_645, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_103 = relay.nn.bias_add(call_102, onnx_Conv_646)
    call_104 = relay.nn.relu(call_90)
    call_105 = relay.nn.relu(call_95)
    call_106 = relay.nn.relu(call_100)
    call_107 = relay.nn.relu(call_103)
    call_109 = relay.concatenate(relay.Tuple([call_104, call_105, call_106, call_107]), axis=1)
    call_110 = relay.nn.conv2d(call_109, onnx_Conv_648, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_111 = relay.nn.bias_add(call_110, onnx_Conv_649)
    call_112 = relay.nn.conv2d(call_109, onnx_Conv_651, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_113 = relay.nn.bias_add(call_112, onnx_Conv_652)
    call_114 = relay.nn.relu(call_113)
    call_115 = relay.nn.conv2d(call_114, onnx_Conv_654, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_116 = relay.nn.bias_add(call_115, onnx_Conv_655)
    call_117 = relay.nn.conv2d(call_109, onnx_Conv_657, padding=[0, 0, 0, 0], channels=24, kernel_size=[1, 1])
    call_118 = relay.nn.bias_add(call_117, onnx_Conv_658)
    call_119 = relay.nn.relu(call_118)
    call_120 = relay.nn.conv2d(call_119, onnx_Conv_660, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_121 = relay.nn.bias_add(call_120, onnx_Conv_661)
    call_122 = relay.nn.max_pool2d(call_109, pool_size=[3, 3], padding=[1, 1, 1, 1], ceil_mode=True)
    call_123 = relay.nn.conv2d(call_122, onnx_Conv_663, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_124 = relay.nn.bias_add(call_123, onnx_Conv_664)
    call_125 = relay.nn.relu(call_111)
    call_126 = relay.nn.relu(call_116)
    call_127 = relay.nn.relu(call_121)
    call_128 = relay.nn.relu(call_124)
    call_130 = relay.concatenate(relay.Tuple([call_125, call_126, call_127, call_128]), axis=1)
    call_131 = relay.nn.conv2d(call_130, onnx_Conv_666, padding=[0, 0, 0, 0], channels=112, kernel_size=[1, 1])
    call_132 = relay.nn.bias_add(call_131, onnx_Conv_667)
    call_133 = relay.nn.conv2d(call_130, onnx_Conv_669, padding=[0, 0, 0, 0], channels=144, kernel_size=[1, 1])
    call_134 = relay.nn.bias_add(call_133, onnx_Conv_670)
    call_135 = relay.nn.relu(call_134)
    call_136 = relay.nn.conv2d(call_135, onnx_Conv_672, padding=[1, 1, 1, 1], channels=288, kernel_size=[3, 3])
    call_137 = relay.nn.bias_add(call_136, onnx_Conv_673)
    call_138 = relay.nn.conv2d(call_130, onnx_Conv_675, padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1])
    call_139 = relay.nn.bias_add(call_138, onnx_Conv_676)
    call_140 = relay.nn.relu(call_139)
    call_141 = relay.nn.conv2d(call_140, onnx_Conv_678, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_142 = relay.nn.bias_add(call_141, onnx_Conv_679)
    call_143 = relay.nn.max_pool2d(call_130, pool_size=[3, 3], padding=[1, 1, 1, 1], ceil_mode=True)
    call_144 = relay.nn.conv2d(call_143, onnx_Conv_681, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_145 = relay.nn.bias_add(call_144, onnx_Conv_682)
    call_146 = relay.nn.relu(call_132)
    call_147 = relay.nn.relu(call_137)
    call_148 = relay.nn.relu(call_142)
    call_149 = relay.nn.relu(call_145)
    call_151 = relay.concatenate(relay.Tuple([call_146, call_147, call_148, call_149]), axis=1)
    call_152 = relay.nn.conv2d(call_151, onnx_Conv_684, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_153 = relay.nn.bias_add(call_152, onnx_Conv_685)
    call_154 = relay.nn.conv2d(call_151, onnx_Conv_687, padding=[0, 0, 0, 0], channels=160, kernel_size=[1, 1])
    call_155 = relay.nn.bias_add(call_154, onnx_Conv_688)
    call_156 = relay.nn.relu(call_155)
    call_157 = relay.nn.conv2d(call_156, onnx_Conv_690, padding=[1, 1, 1, 1], channels=320, kernel_size=[3, 3])
    call_158 = relay.nn.bias_add(call_157, onnx_Conv_691)
    call_159 = relay.nn.conv2d(call_151, onnx_Conv_693, padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1])
    call_160 = relay.nn.bias_add(call_159, onnx_Conv_694)
    call_161 = relay.nn.relu(call_160)
    call_162 = relay.nn.conv2d(call_161, onnx_Conv_696, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_163 = relay.nn.bias_add(call_162, onnx_Conv_697)
    call_164 = relay.nn.max_pool2d(call_151, pool_size=[3, 3], padding=[1, 1, 1, 1], ceil_mode=True)
    call_165 = relay.nn.conv2d(call_164, onnx_Conv_699, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_166 = relay.nn.bias_add(call_165, onnx_Conv_700)
    call_167 = relay.nn.relu(call_153)
    call_168 = relay.nn.relu(call_158)
    call_169 = relay.nn.relu(call_163)
    call_170 = relay.nn.relu(call_166)
    call_172 = relay.concatenate(relay.Tuple([call_167, call_168, call_169, call_170]), axis=1)
    call_173 = relay.nn.max_pool2d(call_172, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0], ceil_mode=True)
    call_174 = relay.nn.conv2d(call_173, onnx_Conv_702, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_175 = relay.nn.bias_add(call_174, onnx_Conv_703)
    call_176 = relay.nn.conv2d(call_173, onnx_Conv_705, padding=[0, 0, 0, 0], channels=160, kernel_size=[1, 1])
    call_177 = relay.nn.bias_add(call_176, onnx_Conv_706)
    call_178 = relay.nn.relu(call_177)
    call_179 = relay.nn.conv2d(call_178, onnx_Conv_708, padding=[1, 1, 1, 1], channels=320, kernel_size=[3, 3])
    call_180 = relay.nn.bias_add(call_179, onnx_Conv_709)
    call_181 = relay.nn.conv2d(call_173, onnx_Conv_711, padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1])
    call_182 = relay.nn.bias_add(call_181, onnx_Conv_712)
    call_183 = relay.nn.relu(call_182)
    call_184 = relay.nn.conv2d(call_183, onnx_Conv_714, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_185 = relay.nn.bias_add(call_184, onnx_Conv_715)
    call_186 = relay.nn.max_pool2d(call_173, pool_size=[3, 3], padding=[1, 1, 1, 1], ceil_mode=True)
    call_187 = relay.nn.conv2d(call_186, onnx_Conv_717, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_188 = relay.nn.bias_add(call_187, onnx_Conv_718)
    call_189 = relay.nn.relu(call_175)
    call_190 = relay.nn.relu(call_180)
    call_191 = relay.nn.relu(call_185)
    call_192 = relay.nn.relu(call_188)
    call_194 = relay.concatenate(relay.Tuple([call_189, call_190, call_191, call_192]), axis=1)
    call_195 = relay.nn.conv2d(call_194, onnx_Conv_720, padding=[0, 0, 0, 0], channels=384, kernel_size=[1, 1])
    call_196 = relay.nn.bias_add(call_195, onnx_Conv_721)
    call_197 = relay.nn.conv2d(call_194, onnx_Conv_723, padding=[0, 0, 0, 0], channels=192, kernel_size=[1, 1])
    call_198 = relay.nn.bias_add(call_197, onnx_Conv_724)
    call_199 = relay.nn.relu(call_198)
    call_200 = relay.nn.conv2d(call_199, onnx_Conv_726, padding=[1, 1, 1, 1], channels=384, kernel_size=[3, 3])
    call_201 = relay.nn.bias_add(call_200, onnx_Conv_727)
    call_202 = relay.nn.conv2d(call_194, onnx_Conv_729, padding=[0, 0, 0, 0], channels=48, kernel_size=[1, 1])
    call_203 = relay.nn.bias_add(call_202, onnx_Conv_730)
    call_204 = relay.nn.relu(call_203)
    call_205 = relay.nn.conv2d(call_204, onnx_Conv_732, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_206 = relay.nn.bias_add(call_205, onnx_Conv_733)
    call_207 = relay.nn.max_pool2d(call_194, pool_size=[3, 3], padding=[1, 1, 1, 1], ceil_mode=True)
    call_208 = relay.nn.conv2d(call_207, onnx_Conv_735, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_209 = relay.nn.bias_add(call_208, onnx_Conv_736)
    call_210 = relay.nn.relu(call_196)
    call_211 = relay.nn.relu(call_201)
    call_212 = relay.nn.relu(call_206)
    call_213 = relay.nn.relu(call_209)
    call_215 = relay.concatenate(relay.Tuple([call_210, call_211, call_212, call_213]), axis=1)
    call_216 = relay.nn.global_avg_pool2d(call_215)
    call_217 = relay.nn.batch_flatten(call_216)
    call_218 = relay.nn.batch_flatten(call_217)
    call_219 = relay.nn.dense(call_218, fc_weight, units=1000)
    call_220 = relay.multiply(relay.const(1, dtype="float32"), fc_bias)
    call_output0 = relay.add(call_219, call_220)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]