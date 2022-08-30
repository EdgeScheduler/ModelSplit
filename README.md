# ModelSplit

Try to Split one model to many smaller models

## Time Test

> shape=(1,3,224,224)

- dell04
  - cpu: 
    - 0.1935865879058838
    - 0.17652320861816406
  - gpu
    - 0.4194180965423584
    - 0.005059957504272461
- P1000 
  - 0.030528545379638672
  - 0.0256803035736084


|                    | dell04-gpu(2080Ti)   | dell04-cpu          | P1000                | 开发板A9 |
| ------------------ | -------------------- | ------------------- | -------------------- | -------- |
| 冷启动：第一次运行 | 0.4194180965423584   | 0.1935865879058838  | 0.030528545379638672 | 4.442520 |
| 第二次+            | 0.005059957504272461 | 0.17652320861816406 | 0.0256803035736084   | 2.276656 |

## Support

only support to parse & split IRModule text without meta[Constant] value

- DL model
  - googlenet
  - resnet50
  - squeezenet
  - vgg19
  - yolov2
- convergence node
  - 2 limits
    - a node has only one next node, and the next node only has one prior node, too.
    - there is no data flow across from priors to next
      - priors: the total of in degree(include convergence)==the total of out degree((exclude convergence))
- split model steps
  - init: get a IRModule text file, with a value shape
  - parse IRModule text, build graph with the relationship of operators, find convergence node points
  - choose one or more convergence node points as the split points of the model
  - split the IRModule text as two or more parts, store as some split-IRModule text and a params json file
    - split-IRModule text need to handle input(add new input & remove (include redundant input/params)
    - params json file record the input & ouput of each split IRModule
  - export python file with split-IRModule text
    - the python file consists of  a function that return RelayExpr, so you can just import this function
    - use `tvm.IRModule.from_expr` to translate RelayExpr to IRModule