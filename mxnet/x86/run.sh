model_names_file='model_names.json'
exe_time='./log/model_exec_time'
config_err='./log/target_config_err'
graph_debug_info='./log/graph_debug_info'

if [ $# -gt 0 ]; then
    model_names_file=$1
fi

if [ $# -gt 1 ]; then
    exe_time=$2
fi

if [ $# -gt 2 ]; then
    config_err=$3
fi

if [ $# -gt 3 ]; then
    graph_debug_info=$4
fi

echo 'start test'
python3 from_mxnet.py $model_names_file  1> $exe_time  $config_err 2> $graph_debug_info
echo 'test finish'
python3 read_result.py $exe_time >result

