export CUDA_HOME=/usr/local/apps/cuda/12.1
export CC=/usr/local/apps/gcc/11.2/bin/gcc
export CXX=/usr/local/apps/gcc/11.2/bin/g++
export CUDA_VISIBLE_DEVICES=0

RUN_NAME=cifar10
HPC_SHARE=/nfs/stak/users/timilsis/hpc-share/i2i
IMB=100

export INCEPTION_PATH=/nfs/stak/users/timilsis/hpc-share/i2i/inception_details/inception_$RUN_NAME

python -W ignore train.py --outdir=$HPC_SHARE/training_runs_$RUN_NAME \
--data=$HPC_SHARE/$RUN_NAME/ --fname=lt_$IMB.json \
--cond=1 --num_classes=10  \
--gpus=1 --seed=69 \
--cfg=auto --mirror=1 --snap=200 \
--metrics=fid_shots,kid_shots,fid50k_full,kid50k_full --cfg=cifar;