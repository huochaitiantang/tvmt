for arg in "$@"
do
    echo $arg
    python3 trt.py --model $arg
done

