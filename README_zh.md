

# PvNet_TensorRT8

​	使用TensorRT实现PvNet部署推理以及后处理，包含CUDA算子加速的RANSAC后处理。

## 环境

1. 测试环境在Jetson嵌入式设备（JetPack 5.2）上，在其他平台上确保安装了显卡驱动、CUDA、TensorRT。
2. 安装第三方库cnpy，链接在[here](https://github.com/rogersce/cnpy.git)

## 安装步骤

1. 下载仓库

```sh
git clone https://github.com/NearianChen/PvNet_TensorRT8.git
```

2. 安装cnpy

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

3. 编译

```shell
cd PvNet_TensorRT8
mkdir build
cd build
cmake ..
make
```



