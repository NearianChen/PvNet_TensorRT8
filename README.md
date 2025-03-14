

# PvNet_TensorRT8

​	PvNet deployment inference using TensorRT and post-processing, including RANSAC post-processing accelerated by CUDA operators.

## Environment

1. test environment on Jetson embedded device (JetPack 5.2), on other platforms make sure that graphics drivers, CUDA, TensorRT are installed.
2. Install the third-party repo cnpy, the link is at [here](https://github.com/rogersce/cnpy.git)

## Install

1. git clone repo

```sh
git clone https://github.com/NearianChen/PvNet_TensorRT8.git
```

2. install cnpy

```shell
mkdir thirdparty 
cd thirdparty
git clone https://github.com/rogersce/cnpy.git
cd cnpy
mkdir build
cd build
cmake ..
make
make install
```

3. make

```shell
cd PvNet_TensorRT8
mkdir build
cd build
cmake ..
make
```



