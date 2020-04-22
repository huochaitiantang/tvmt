示例依赖修改过的 tvm 代码，修改涉及 c++ 代码，因此需要重新编译，必要时需要重新安装。

依赖 tvm 分支为 [tvmd/eerrrec](https://github.com/huochaitiantang/tvmd/tree/errec)

## 接口

* `tvm.tvmt.log_to_sqlite(filename)`

    获取将一轮编译、运行的结果保存至 `filename` 的 callback function，需要传递给 `tuner.tune(...)` 的 `callback` 参数。

## tvm 内部改动

* 增加并使用 `tvmt.report_json` packed_function

    通过 `tvm._ffi.register_func("tvmt.report_json")` 注册一个接受 json 字符串或者 python 对象的函数，这个函数可以在 c++ 和 python 端被调用，在 tvm 中多出模糊化报错信息的地方，都可以加上这个函数。函数将报错信息保存到全局变量中（保证该全局变量中的信息全部都指向同一任务实例）。

* 增加 sqlite log 方式，保存每一个任务实例。设计的数据库为
    * logs

        保存和一般 log 中基本相同的信息，提出部分字段为表头（如 target、cost），以方便查询。

        | *id* | task_name | target | *workload* | config | err_no | cost | err_text |
        |------|-----------|--------|------------|--------|--------|------|----------|
        | 与 Config 实体在原一维空间的位置对应 | `tvm.autotvm.task.create` 的第一个参数 | 比如 cuda，和一般 log 中一样 | 序列化的 task 信息，主要包含 task 对应 function 的参数、Tensor shape、dtype 等信息。和一般 log 中一样 | 包含了一个 config 的完整信息，比如各个 knob 的取值，和一般 log 中一样 | 错误编号，参考 #1 中的总结 | 去掉最快、最慢后的平均 cost | 报错信息，和一般 log 中一样（不一定完整） |

    * kvs

        保存通过 `tvmt.report_json` packed_function 所保存的额外信息，保存形式为 k-v pair。

        | *id* | log_id | log_workload | k | v |
        |------|--------|--------------|---|---|
        | 编号，不重要 | foreign key -> `logs.id` | foreign key -> `logs.workload` | 字段名 | 字段值 |