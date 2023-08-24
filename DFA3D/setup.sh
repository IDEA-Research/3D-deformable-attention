debug_or_not=$1
if [ $debug_or_not = 1 ]
then
    CUDA_ARGS="-Xcompiler,-fPIC,-g,-G"
else
    CUDA_ARGS=""
fi
export CUDA_ARGS
python setup.py install