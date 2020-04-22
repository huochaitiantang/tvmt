import os
import tvm
import tvm.tvmt
import tvm.autotvm
import topi.cuda.conv2d_int8

import topi


def main():

    i = tvm.te.placeholder([8, 4, 128, 128], dtype='int8', name='name')
    k = tvm.te.placeholder([4, 4, 64, 64], dtype='int8', name='name')
    with tvm.target.cuda():
        g = topi.cuda.conv2d_NCHWc_int8(i, k, 1, 0, 1, 'NCHW', 'int32')
        s = topi.cuda.schedule_conv2d_NCHWc_int8(g)

    t = tvm.autotvm.task.create(
        'conv2d_NCHWc_int8.cuda', (i, k, 1, 0, 1, 'NCHW', 'int32'), target='cuda')
    # builder = tvm.autotvm.LocalBuilder(n_parallel=1)
    builder = tvm.autotvm.LocalBuilder()
    runner = tvm.autotvm.LocalRunner(repeat=8, min_repeat_ms=100, timeout=30)
    # uncomment these two lines while debugging
    # builder.executor = tvm.autotvm.measure.measure_methods.LocalExecutor(timeout=10, do_fork=False)
    # runner.executor = tvm.autotvm.measure.measure_methods.LocalExecutor(timeout=10, do_fork=False)
    measure_option = tvm.autotvm.measure_option(
        builder=builder,
        runner=runner
    )

    tuner = tvm.autotvm.tuner.XGBTuner(t)

    logfile = 'test_single_kernel.log'
    try:
        print('loading')
        tuner.load_history(tvm.autotvm.record.load_from_file(logfile))
        print('load done')
    except Exception as e:
        print('Load log failed', e)
    b = 0

    def rp(*args):
        nonlocal b
        print("batch:", b)
        b += 1
    tuner.tune(n_trial=1000,
               measure_option=measure_option,
               callbacks=[
                   tvm.autotvm.callback.log_to_file(logfile),
                   tvm.tvmt.log_to_sqlite(logfile+'.db'),
                   rp,
               ])


main()
