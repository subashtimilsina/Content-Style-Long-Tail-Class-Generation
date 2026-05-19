export CUDA_HOME=/usr/local/apps/cuda/12.1
export CC=/usr/local/apps/gcc/11.2/bin/gcc
export CXX=/usr/local/apps/gcc/11.2/bin/g++
export CUDA_VISIBLE_DEVICES=0

SEEDS=00000,11111,22222,33333,44444,55555,66666,77777,88888,99999

NC=102
PKL_FILE=/nfs/stak/users/timilsis/Projects/hpc_share/i2i/training_runs_flowers/00035--cond-mirror-lt_100.json-auto/network-snapshot-008064.pkl

python generate_c_s.py --outdir=./results/ --seeds=$SEEDS --all_class=$NC --network=$PKL_FILE
