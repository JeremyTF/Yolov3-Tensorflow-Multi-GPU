export PYTHONPATH=/home/lei/python:/home/lei/python/lib64/python2.7/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
export PATH=$PATH:/usr/local/cuda-10.0/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-10.0
CUDA_VISIBLE_DEVICES=1 python evaluate.py
