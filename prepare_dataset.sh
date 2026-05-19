python lt_dataset.py --fn /nfs/stak/users/timilsis/hpc-share/i2i/af/dataset.json \
 --imf 25 --shot-labels --dname animals

python dataset_tool.py --source=/nfs/stak/users/timilsis/hpc-share/i2i/raw_flowers \
 --dest=/nfs/stak/users/timilsis/hpc-share/i2i/flowers --transform=center-crop --width=128 --height=128

python lt_dataset.py --fn /nfs/stak/users/timilsis/hpc-share/i2i/flowers/dataset.json \
 --imf 100 --shot-labels --dname flowers

python dataset_tool.py --source=/nfs/stak/users/timilsis/hpc-share/i2i/lsun_subset \
 --dest=/nfs/stak/users/timilsis/hpc-share/i2i/lsun --transform=center-crop --width=128 --height=128

python lt_dataset.py --fn /nfs/stak/users/timilsis/hpc-share/i2i/lsun/dataset.json \
 --imf 1000 --shot-labels --dname lsun

python dataset_tool.py --source=/nfs/stak/users/timilsis/hpc-share/i2i/cifar-10-python.tar.gz \
 --dest=/nfs/stak/users/timilsis/hpc-share/i2i/cifar10 --transform=center-crop --width=32 --height=32

python lt_dataset.py --fn /nfs/stak/users/timilsis/hpc-share/i2i/cifar10/dataset.json \
 --imf 100 --shot-labels --dname cifar10


python dataset_tool.py --source=/nfs/stak/users/timilsis/hpc-share/i2i/cifar-100-python.tar.gz \
 --dest=/nfs/stak/users/timilsis/hpc-share/i2i/cifar100 --transform=center-crop --width=32 --height=32

python lt_dataset.py --fn /nfs/stak/users/timilsis/hpc-share/i2i/cifar100/dataset.json \
 --imf 100 --shot-labels --dname cifar100