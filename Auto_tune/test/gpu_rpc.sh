# python3 -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190 
# python3 -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=V100
# python3 -m tvm.exec.query_rpc_tracker --port 9190

python3 -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190  &
sleep 1
python3 -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=V100 &
sleep 1
python3 -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190
