wget https://vcla.stat.ucla.edu/people/zhangzhang-si/HiT/AnimalFace.zip
unzip AnimalFace.zip -d ./
rm -rf Image/Natural
rm AnimalFace.zip

wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzf cifar-100-python.tar.gz
rm cifar-100-python.tar.gz

wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz

#Flowers
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
tar -xzf 102flowers.tgz
rm 102flowers.tgz

#LSUN
wget http://lsun.csail.mit.edu/download/bedroom_train_lmdb.zip
unzip bedroom_train_lmdb.zip -d ./
rm bedroom_train_lmdb.zip
