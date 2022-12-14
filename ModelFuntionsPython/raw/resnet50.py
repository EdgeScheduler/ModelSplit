# this file is create by program.def Resnet50(pre_input=None):    import tvm    import numpy as np    from tvm import relay    data = pre_input if pre_input is not None else relay.var("data", shape=(15, 3, 224, 224), dtype="float32")    fc_weight = relay.var("fc.weight", shape=(1000, 2048), dtype="float32")    fc_bias = relay.var("fc.bias", shape=(1000, ), dtype="float32")    onnx_Conv_497 = relay.var("onnx_Conv_497", shape=(64, 3, 7, 7), dtype="float32")    onnx_Conv_498 = relay.var("onnx_Conv_498", shape=(64, ), dtype="float32")    onnx_Conv_500 = relay.var("onnx_Conv_500", shape=(64, 64, 1, 1), dtype="float32")    onnx_Conv_501 = relay.var("onnx_Conv_501", shape=(64, ), dtype="float32")    onnx_Conv_503 = relay.var("onnx_Conv_503", shape=(64, 64, 3, 3), dtype="float32")    onnx_Conv_504 = relay.var("onnx_Conv_504", shape=(64, ), dtype="float32")    onnx_Conv_506 = relay.var("onnx_Conv_506", shape=(256, 64, 1, 1), dtype="float32")    onnx_Conv_507 = relay.var("onnx_Conv_507", shape=(256, ), dtype="float32")    onnx_Conv_509 = relay.var("onnx_Conv_509", shape=(256, 64, 1, 1), dtype="float32")    onnx_Conv_510 = relay.var("onnx_Conv_510", shape=(256, ), dtype="float32")    onnx_Conv_512 = relay.var("onnx_Conv_512", shape=(64, 256, 1, 1), dtype="float32")    onnx_Conv_513 = relay.var("onnx_Conv_513", shape=(64, ), dtype="float32")    onnx_Conv_515 = relay.var("onnx_Conv_515", shape=(64, 64, 3, 3), dtype="float32")    onnx_Conv_516 = relay.var("onnx_Conv_516", shape=(64, ), dtype="float32")    onnx_Conv_518 = relay.var("onnx_Conv_518", shape=(256, 64, 1, 1), dtype="float32")    onnx_Conv_519 = relay.var("onnx_Conv_519", shape=(256, ), dtype="float32")    onnx_Conv_521 = relay.var("onnx_Conv_521", shape=(64, 256, 1, 1), dtype="float32")    onnx_Conv_522 = relay.var("onnx_Conv_522", shape=(64, ), dtype="float32")    onnx_Conv_524 = relay.var("onnx_Conv_524", shape=(64, 64, 3, 3), dtype="float32")    onnx_Conv_525 = relay.var("onnx_Conv_525", shape=(64, ), dtype="float32")    onnx_Conv_527 = relay.var("onnx_Conv_527", shape=(256, 64, 1, 1), dtype="float32")    onnx_Conv_528 = relay.var("onnx_Conv_528", shape=(256, ), dtype="float32")    onnx_Conv_530 = relay.var("onnx_Conv_530", shape=(128, 256, 1, 1), dtype="float32")    onnx_Conv_531 = relay.var("onnx_Conv_531", shape=(128, ), dtype="float32")    onnx_Conv_533 = relay.var("onnx_Conv_533", shape=(128, 128, 3, 3), dtype="float32")    onnx_Conv_534 = relay.var("onnx_Conv_534", shape=(128, ), dtype="float32")    onnx_Conv_536 = relay.var("onnx_Conv_536", shape=(512, 128, 1, 1), dtype="float32")    onnx_Conv_537 = relay.var("onnx_Conv_537", shape=(512, ), dtype="float32")    onnx_Conv_539 = relay.var("onnx_Conv_539", shape=(512, 256, 1, 1), dtype="float32")    onnx_Conv_540 = relay.var("onnx_Conv_540", shape=(512, ), dtype="float32")    onnx_Conv_542 = relay.var("onnx_Conv_542", shape=(128, 512, 1, 1), dtype="float32")    onnx_Conv_543 = relay.var("onnx_Conv_543", shape=(128, ), dtype="float32")    onnx_Conv_545 = relay.var("onnx_Conv_545", shape=(128, 128, 3, 3), dtype="float32")    onnx_Conv_546 = relay.var("onnx_Conv_546", shape=(128, ), dtype="float32")    onnx_Conv_548 = relay.var("onnx_Conv_548", shape=(512, 128, 1, 1), dtype="float32")    onnx_Conv_549 = relay.var("onnx_Conv_549", shape=(512, ), dtype="float32")    onnx_Conv_551 = relay.var("onnx_Conv_551", shape=(128, 512, 1, 1), dtype="float32")    onnx_Conv_552 = relay.var("onnx_Conv_552", shape=(128, ), dtype="float32")    onnx_Conv_554 = relay.var("onnx_Conv_554", shape=(128, 128, 3, 3), dtype="float32")    onnx_Conv_555 = relay.var("onnx_Conv_555", shape=(128, ), dtype="float32")    onnx_Conv_557 = relay.var("onnx_Conv_557", shape=(512, 128, 1, 1), dtype="float32")    onnx_Conv_558 = relay.var("onnx_Conv_558", shape=(512, ), dtype="float32")    onnx_Conv_560 = relay.var("onnx_Conv_560", shape=(128, 512, 1, 1), dtype="float32")    onnx_Conv_561 = relay.var("onnx_Conv_561", shape=(128, ), dtype="float32")    onnx_Conv_563 = relay.var("onnx_Conv_563", shape=(128, 128, 3, 3), dtype="float32")    onnx_Conv_564 = relay.var("onnx_Conv_564", shape=(128, ), dtype="float32")    onnx_Conv_566 = relay.var("onnx_Conv_566", shape=(512, 128, 1, 1), dtype="float32")    onnx_Conv_567 = relay.var("onnx_Conv_567", shape=(512, ), dtype="float32")    onnx_Conv_569 = relay.var("onnx_Conv_569", shape=(256, 512, 1, 1), dtype="float32")    onnx_Conv_570 = relay.var("onnx_Conv_570", shape=(256, ), dtype="float32")    onnx_Conv_572 = relay.var("onnx_Conv_572", shape=(256, 256, 3, 3), dtype="float32")    onnx_Conv_573 = relay.var("onnx_Conv_573", shape=(256, ), dtype="float32")    onnx_Conv_575 = relay.var("onnx_Conv_575", shape=(1024, 256, 1, 1), dtype="float32")    onnx_Conv_576 = relay.var("onnx_Conv_576", shape=(1024, ), dtype="float32")    onnx_Conv_578 = relay.var("onnx_Conv_578", shape=(1024, 512, 1, 1), dtype="float32")    onnx_Conv_579 = relay.var("onnx_Conv_579", shape=(1024, ), dtype="float32")    onnx_Conv_581 = relay.var("onnx_Conv_581", shape=(256, 1024, 1, 1), dtype="float32")    onnx_Conv_582 = relay.var("onnx_Conv_582", shape=(256, ), dtype="float32")    onnx_Conv_584 = relay.var("onnx_Conv_584", shape=(256, 256, 3, 3), dtype="float32")    onnx_Conv_585 = relay.var("onnx_Conv_585", shape=(256, ), dtype="float32")    onnx_Conv_587 = relay.var("onnx_Conv_587", shape=(1024, 256, 1, 1), dtype="float32")    onnx_Conv_588 = relay.var("onnx_Conv_588", shape=(1024, ), dtype="float32")    onnx_Conv_590 = relay.var("onnx_Conv_590", shape=(256, 1024, 1, 1), dtype="float32")    onnx_Conv_591 = relay.var("onnx_Conv_591", shape=(256, ), dtype="float32")    onnx_Conv_593 = relay.var("onnx_Conv_593", shape=(256, 256, 3, 3), dtype="float32")    onnx_Conv_594 = relay.var("onnx_Conv_594", shape=(256, ), dtype="float32")    onnx_Conv_596 = relay.var("onnx_Conv_596", shape=(1024, 256, 1, 1), dtype="float32")    onnx_Conv_597 = relay.var("onnx_Conv_597", shape=(1024, ), dtype="float32")    onnx_Conv_599 = relay.var("onnx_Conv_599", shape=(256, 1024, 1, 1), dtype="float32")    onnx_Conv_600 = relay.var("onnx_Conv_600", shape=(256, ), dtype="float32")    onnx_Conv_602 = relay.var("onnx_Conv_602", shape=(256, 256, 3, 3), dtype="float32")    onnx_Conv_603 = relay.var("onnx_Conv_603", shape=(256, ), dtype="float32")    onnx_Conv_605 = relay.var("onnx_Conv_605", shape=(1024, 256, 1, 1), dtype="float32")    onnx_Conv_606 = relay.var("onnx_Conv_606", shape=(1024, ), dtype="float32")    onnx_Conv_608 = relay.var("onnx_Conv_608", shape=(256, 1024, 1, 1), dtype="float32")    onnx_Conv_609 = relay.var("onnx_Conv_609", shape=(256, ), dtype="float32")    onnx_Conv_611 = relay.var("onnx_Conv_611", shape=(256, 256, 3, 3), dtype="float32")    onnx_Conv_612 = relay.var("onnx_Conv_612", shape=(256, ), dtype="float32")    onnx_Conv_614 = relay.var("onnx_Conv_614", shape=(1024, 256, 1, 1), dtype="float32")    onnx_Conv_615 = relay.var("onnx_Conv_615", shape=(1024, ), dtype="float32")    onnx_Conv_617 = relay.var("onnx_Conv_617", shape=(256, 1024, 1, 1), dtype="float32")    onnx_Conv_618 = relay.var("onnx_Conv_618", shape=(256, ), dtype="float32")    onnx_Conv_620 = relay.var("onnx_Conv_620", shape=(256, 256, 3, 3), dtype="float32")    onnx_Conv_621 = relay.var("onnx_Conv_621", shape=(256, ), dtype="float32")    onnx_Conv_623 = relay.var("onnx_Conv_623", shape=(1024, 256, 1, 1), dtype="float32")    onnx_Conv_624 = relay.var("onnx_Conv_624", shape=(1024, ), dtype="float32")    onnx_Conv_626 = relay.var("onnx_Conv_626", shape=(512, 1024, 1, 1), dtype="float32")    onnx_Conv_627 = relay.var("onnx_Conv_627", shape=(512, ), dtype="float32")    onnx_Conv_629 = relay.var("onnx_Conv_629", shape=(512, 512, 3, 3), dtype="float32")    onnx_Conv_630 = relay.var("onnx_Conv_630", shape=(512, ), dtype="float32")    onnx_Conv_632 = relay.var("onnx_Conv_632", shape=(2048, 512, 1, 1), dtype="float32")    onnx_Conv_633 = relay.var("onnx_Conv_633", shape=(2048, ), dtype="float32")    onnx_Conv_635 = relay.var("onnx_Conv_635", shape=(2048, 1024, 1, 1), dtype="float32")    onnx_Conv_636 = relay.var("onnx_Conv_636", shape=(2048, ), dtype="float32")    onnx_Conv_638 = relay.var("onnx_Conv_638", shape=(512, 2048, 1, 1), dtype="float32")    onnx_Conv_639 = relay.var("onnx_Conv_639", shape=(512, ), dtype="float32")    onnx_Conv_641 = relay.var("onnx_Conv_641", shape=(512, 512, 3, 3), dtype="float32")    onnx_Conv_642 = relay.var("onnx_Conv_642", shape=(512, ), dtype="float32")    onnx_Conv_644 = relay.var("onnx_Conv_644", shape=(2048, 512, 1, 1), dtype="float32")    onnx_Conv_645 = relay.var("onnx_Conv_645", shape=(2048, ), dtype="float32")    onnx_Conv_647 = relay.var("onnx_Conv_647", shape=(512, 2048, 1, 1), dtype="float32")    onnx_Conv_648 = relay.var("onnx_Conv_648", shape=(512, ), dtype="float32")    onnx_Conv_650 = relay.var("onnx_Conv_650", shape=(512, 512, 3, 3), dtype="float32")    onnx_Conv_651 = relay.var("onnx_Conv_651", shape=(512, ), dtype="float32")    onnx_Conv_653 = relay.var("onnx_Conv_653", shape=(2048, 512, 1, 1), dtype="float32")    onnx_Conv_654 = relay.var("onnx_Conv_654", shape=(2048, ), dtype="float32")    call_0 = relay.nn.conv2d(data, onnx_Conv_497, strides=[2, 2], padding=[3, 3, 3, 3], channels=64, kernel_size=[7, 7])
    call_1 = relay.nn.bias_add(call_0, onnx_Conv_498)
    call_2 = relay.nn.relu(call_1)
    call_3 = relay.nn.max_pool2d(call_2, pool_size=[3, 3], strides=[2, 2], padding=[1, 1, 1, 1])
    call_4 = relay.nn.conv2d(call_3, onnx_Conv_500, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_5 = relay.nn.bias_add(call_4, onnx_Conv_501)
    call_6 = relay.nn.relu(call_5)
    call_7 = relay.nn.conv2d(call_6, onnx_Conv_503, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_8 = relay.nn.bias_add(call_7, onnx_Conv_504)
    call_9 = relay.nn.relu(call_8)
    call_10 = relay.nn.conv2d(call_9, onnx_Conv_506, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_11 = relay.nn.conv2d(call_3, onnx_Conv_509, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_12 = relay.nn.bias_add(call_10, onnx_Conv_507)
    call_13 = relay.nn.bias_add(call_11, onnx_Conv_510)
    call_14 = relay.add(call_12, call_13)
    call_15 = relay.nn.relu(call_14)
    call_16 = relay.nn.conv2d(call_15, onnx_Conv_512, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_17 = relay.nn.bias_add(call_16, onnx_Conv_513)
    call_18 = relay.nn.relu(call_17)
    call_19 = relay.nn.conv2d(call_18, onnx_Conv_515, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_20 = relay.nn.bias_add(call_19, onnx_Conv_516)
    call_21 = relay.nn.relu(call_20)
    call_22 = relay.nn.conv2d(call_21, onnx_Conv_518, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_23 = relay.nn.bias_add(call_22, onnx_Conv_519)
    call_24 = relay.add(call_23, call_15)
    call_25 = relay.nn.relu(call_24)
    call_26 = relay.nn.conv2d(call_25, onnx_Conv_521, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1])
    call_27 = relay.nn.bias_add(call_26, onnx_Conv_522)
    call_28 = relay.nn.relu(call_27)
    call_29 = relay.nn.conv2d(call_28, onnx_Conv_524, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
    call_30 = relay.nn.bias_add(call_29, onnx_Conv_525)
    call_31 = relay.nn.relu(call_30)
    call_32 = relay.nn.conv2d(call_31, onnx_Conv_527, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_33 = relay.nn.bias_add(call_32, onnx_Conv_528)
    call_34 = relay.add(call_33, call_25)
    call_35 = relay.nn.relu(call_34)
    call_36 = relay.nn.conv2d(call_35, onnx_Conv_530, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_37 = relay.nn.bias_add(call_36, onnx_Conv_531)
    call_38 = relay.nn.relu(call_37)
    call_39 = relay.nn.conv2d(call_38, onnx_Conv_533, strides=[2, 2], padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_40 = relay.nn.bias_add(call_39, onnx_Conv_534)
    call_41 = relay.nn.relu(call_40)
    call_42 = relay.nn.conv2d(call_41, onnx_Conv_536, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1])
    call_43 = relay.nn.conv2d(call_35, onnx_Conv_539, strides=[2, 2], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1])
    call_44 = relay.nn.bias_add(call_42, onnx_Conv_537)
    call_45 = relay.nn.bias_add(call_43, onnx_Conv_540)
    call_46 = relay.add(call_44, call_45)
    call_47 = relay.nn.relu(call_46)
    call_48 = relay.nn.conv2d(call_47, onnx_Conv_542, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_49 = relay.nn.bias_add(call_48, onnx_Conv_543)
    call_50 = relay.nn.relu(call_49)
    call_51 = relay.nn.conv2d(call_50, onnx_Conv_545, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_52 = relay.nn.bias_add(call_51, onnx_Conv_546)
    call_53 = relay.nn.relu(call_52)
    call_54 = relay.nn.conv2d(call_53, onnx_Conv_548, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1])
    call_55 = relay.nn.bias_add(call_54, onnx_Conv_549)
    call_56 = relay.add(call_55, call_47)
    call_57 = relay.nn.relu(call_56)
    call_58 = relay.nn.conv2d(call_57, onnx_Conv_551, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_59 = relay.nn.bias_add(call_58, onnx_Conv_552)
    call_60 = relay.nn.relu(call_59)
    call_61 = relay.nn.conv2d(call_60, onnx_Conv_554, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_62 = relay.nn.bias_add(call_61, onnx_Conv_555)
    call_63 = relay.nn.relu(call_62)
    call_64 = relay.nn.conv2d(call_63, onnx_Conv_557, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1])
    call_65 = relay.nn.bias_add(call_64, onnx_Conv_558)
    call_66 = relay.add(call_65, call_57)
    call_67 = relay.nn.relu(call_66)
    call_68 = relay.nn.conv2d(call_67, onnx_Conv_560, padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1])
    call_69 = relay.nn.bias_add(call_68, onnx_Conv_561)
    call_70 = relay.nn.relu(call_69)
    call_71 = relay.nn.conv2d(call_70, onnx_Conv_563, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3])
    call_72 = relay.nn.bias_add(call_71, onnx_Conv_564)
    call_73 = relay.nn.relu(call_72)
    call_74 = relay.nn.conv2d(call_73, onnx_Conv_566, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1])
    call_75 = relay.nn.bias_add(call_74, onnx_Conv_567)
    call_76 = relay.add(call_75, call_67)
    call_77 = relay.nn.relu(call_76)
    call_78 = relay.nn.conv2d(call_77, onnx_Conv_569, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_79 = relay.nn.bias_add(call_78, onnx_Conv_570)
    call_80 = relay.nn.relu(call_79)
    call_81 = relay.nn.conv2d(call_80, onnx_Conv_572, strides=[2, 2], padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_82 = relay.nn.bias_add(call_81, onnx_Conv_573)
    call_83 = relay.nn.relu(call_82)
    call_84 = relay.nn.conv2d(call_83, onnx_Conv_575, padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1])
    call_85 = relay.nn.conv2d(call_77, onnx_Conv_578, strides=[2, 2], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1])
    call_86 = relay.nn.bias_add(call_84, onnx_Conv_576)
    call_87 = relay.nn.bias_add(call_85, onnx_Conv_579)
    call_88 = relay.add(call_86, call_87)
    call_89 = relay.nn.relu(call_88)
    call_90 = relay.nn.conv2d(call_89, onnx_Conv_581, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_91 = relay.nn.bias_add(call_90, onnx_Conv_582)
    call_92 = relay.nn.relu(call_91)
    call_93 = relay.nn.conv2d(call_92, onnx_Conv_584, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_94 = relay.nn.bias_add(call_93, onnx_Conv_585)
    call_95 = relay.nn.relu(call_94)
    call_96 = relay.nn.conv2d(call_95, onnx_Conv_587, padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1])
    call_97 = relay.nn.bias_add(call_96, onnx_Conv_588)
    call_98 = relay.add(call_97, call_89)
    call_99 = relay.nn.relu(call_98)
    call_100 = relay.nn.conv2d(call_99, onnx_Conv_590, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_101 = relay.nn.bias_add(call_100, onnx_Conv_591)
    call_102 = relay.nn.relu(call_101)
    call_103 = relay.nn.conv2d(call_102, onnx_Conv_593, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_104 = relay.nn.bias_add(call_103, onnx_Conv_594)
    call_105 = relay.nn.relu(call_104)
    call_106 = relay.nn.conv2d(call_105, onnx_Conv_596, padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1])
    call_107 = relay.nn.bias_add(call_106, onnx_Conv_597)
    call_108 = relay.add(call_107, call_99)
    call_109 = relay.nn.relu(call_108)
    call_110 = relay.nn.conv2d(call_109, onnx_Conv_599, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_111 = relay.nn.bias_add(call_110, onnx_Conv_600)
    call_112 = relay.nn.relu(call_111)
    call_113 = relay.nn.conv2d(call_112, onnx_Conv_602, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_114 = relay.nn.bias_add(call_113, onnx_Conv_603)
    call_115 = relay.nn.relu(call_114)
    call_116 = relay.nn.conv2d(call_115, onnx_Conv_605, padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1])
    call_117 = relay.nn.bias_add(call_116, onnx_Conv_606)
    call_118 = relay.add(call_117, call_109)
    call_119 = relay.nn.relu(call_118)
    call_120 = relay.nn.conv2d(call_119, onnx_Conv_608, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_121 = relay.nn.bias_add(call_120, onnx_Conv_609)
    call_122 = relay.nn.relu(call_121)
    call_123 = relay.nn.conv2d(call_122, onnx_Conv_611, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_124 = relay.nn.bias_add(call_123, onnx_Conv_612)
    call_125 = relay.nn.relu(call_124)
    call_126 = relay.nn.conv2d(call_125, onnx_Conv_614, padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1])
    call_127 = relay.nn.bias_add(call_126, onnx_Conv_615)
    call_128 = relay.add(call_127, call_119)
    call_129 = relay.nn.relu(call_128)
    call_130 = relay.nn.conv2d(call_129, onnx_Conv_617, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1])
    call_131 = relay.nn.bias_add(call_130, onnx_Conv_618)
    call_132 = relay.nn.relu(call_131)
    call_133 = relay.nn.conv2d(call_132, onnx_Conv_620, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3])
    call_134 = relay.nn.bias_add(call_133, onnx_Conv_621)
    call_135 = relay.nn.relu(call_134)
    call_136 = relay.nn.conv2d(call_135, onnx_Conv_623, padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1])
    call_137 = relay.nn.bias_add(call_136, onnx_Conv_624)
    call_138 = relay.add(call_137, call_129)
    call_139 = relay.nn.relu(call_138)
    call_140 = relay.nn.conv2d(call_139, onnx_Conv_626, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1])
    call_141 = relay.nn.bias_add(call_140, onnx_Conv_627)
    call_142 = relay.nn.relu(call_141)
    call_143 = relay.nn.conv2d(call_142, onnx_Conv_629, strides=[2, 2], padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_144 = relay.nn.bias_add(call_143, onnx_Conv_630)
    call_145 = relay.nn.relu(call_144)
    call_146 = relay.nn.conv2d(call_145, onnx_Conv_632, padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1])
    call_147 = relay.nn.conv2d(call_139, onnx_Conv_635, strides=[2, 2], padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1])
    call_148 = relay.nn.bias_add(call_146, onnx_Conv_633)
    call_149 = relay.nn.bias_add(call_147, onnx_Conv_636)
    call_150 = relay.add(call_148, call_149)
    call_151 = relay.nn.relu(call_150)
    call_152 = relay.nn.conv2d(call_151, onnx_Conv_638, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1])
    call_153 = relay.nn.bias_add(call_152, onnx_Conv_639)
    call_154 = relay.nn.relu(call_153)
    call_155 = relay.nn.conv2d(call_154, onnx_Conv_641, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_156 = relay.nn.bias_add(call_155, onnx_Conv_642)
    call_157 = relay.nn.relu(call_156)
    call_158 = relay.nn.conv2d(call_157, onnx_Conv_644, padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1])
    call_159 = relay.nn.bias_add(call_158, onnx_Conv_645)
    call_160 = relay.add(call_159, call_151)
    call_161 = relay.nn.relu(call_160)
    call_162 = relay.nn.conv2d(call_161, onnx_Conv_647, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1])
    call_163 = relay.nn.bias_add(call_162, onnx_Conv_648)
    call_164 = relay.nn.relu(call_163)
    call_165 = relay.nn.conv2d(call_164, onnx_Conv_650, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
    call_166 = relay.nn.bias_add(call_165, onnx_Conv_651)
    call_167 = relay.nn.relu(call_166)
    call_168 = relay.nn.conv2d(call_167, onnx_Conv_653, padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1])
    call_169 = relay.nn.bias_add(call_168, onnx_Conv_654)
    call_170 = relay.add(call_169, call_161)
    call_171 = relay.nn.relu(call_170)
    call_172 = relay.nn.global_avg_pool2d(call_171)
    call_173 = relay.nn.batch_flatten(call_172)
    call_174 = relay.nn.batch_flatten(call_173)
    call_175 = relay.nn.dense(call_174, fc_weight, units=1000)
    call_176 = relay.multiply(relay.const(1, dtype="float32"), fc_bias)
    call_output0 = relay.add(call_175, call_176)
    return call_output0 if not isinstance(call_output0,tvm.relay.expr.TupleWrapper) else call_output0[0]