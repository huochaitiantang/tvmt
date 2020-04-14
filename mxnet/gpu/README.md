### 测试环境

TVM commit: 0.6(c6f8c23)



CUDA版本: 10.0（V100）



---

### 简单测试

使用`mxnet.gluon.model_zoo.vision.get_model()`来获得网络模型。然后通过relay，build，runtime.create获得mod，最后使用`time_evaluator()`来进行测试

涉及文件：mxnet_evaluate.py



mxnet_evaluate.py 包括如下参数：

| 参数             | 说明                                                         | 备注             |
| :--------------- | :----------------------------------------------------------- | :--------------- |
| model_names_file | 测试模型名称所在文件，以json格式存储，key为“model_names”     | 文件格式为`json` |
| exec_time_info   | 计算时间输出文件，其包括所有的标准输出                       |                  |
| config_err       | python代码的warning和错误输出文件，主要内容为形如“can’t find config for target:…..” |                  |
| target           | 设置tvm.build中的```target```                                | 默认为`cuda`     |
| ctx_dev          | 设置module的运行设备                                         | 默认为`gpu`      |
| ctx_no           | 设置module运行设备的编号                                     | 默认为`0`        |

运行方式如下所示

`python3 mxnet_evaluate.py --ctx_no 1`

代表其它参数为默认参数，运行设备的编号为1号GPU上
