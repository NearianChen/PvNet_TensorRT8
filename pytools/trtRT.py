#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
from PIL import Image
import numpy as np
import tensorrt as trt
import time

def run_save2npy(engine_file_path):
    demo_images = 'data/demo_cat'

    # 初始化均值和标准差
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    # TensorRT 初始化
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    def load_engine(engine_file_path):
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(engine):
        h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
        h_output_seg = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
        h_output_ver = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(2)), dtype=np.float32)
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output_seg = cuda.mem_alloc(h_output_seg.nbytes)
        d_output_ver = cuda.mem_alloc(h_output_ver.nbytes)
        stream = cuda.Stream()
        return h_input, h_output_seg, h_output_ver, d_input, d_output_seg, d_output_ver, stream

    def infer_tensorrt(context, h_input, h_output_seg, h_output_ver, d_input, d_output_seg, d_output_ver, stream, input_data):
        np.copyto(h_input, input_data.ravel())
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output_seg), int(d_output_ver)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output_seg, d_output_seg, stream)
        cuda.memcpy_dtoh_async(h_output_ver, d_output_ver, stream)
        stream.synchronize()
        return h_output_seg, h_output_ver

    # 加载 TensorRT 引擎和分配缓冲区
    engine = load_engine(engine_file_path)
    context = engine.create_execution_context()
            
    h_input, h_output_seg, h_output_ver, d_input, d_output_seg, d_output_ver, stream = allocate_buffers(engine)
    output_dir_seg = 'npy_files/trt_npy_v2/seg_npy'
    output_dir_ver = 'npy_files/trt_npy_v2/ver_npy'
    # 遍历文件夹中的所有图片
    for filename in os.listdir(demo_images):
        img_path = os.path.join(demo_images, filename)
        img_path = "data/demo_cat/000032.jpg"
        demo_image = np.array(Image.open(img_path)).astype(np.float32)
        inp = (((demo_image / 255.) - mean) / std).transpose(2, 0, 1).astype(np.float32)
        inp_trt = inp[None, :]   # TensorRT 输入
        inp_trt.tofile("python_input.bin")

        start_time = time.time()
        # TensorRT 推理
        seg_pred_trt, ver_pred_trt = infer_tensorrt(
            context, h_input, h_output_seg, h_output_ver,
            d_input, d_output_seg, d_output_ver, 
            stream, inp_trt
        )
        # 记录推理结束时间
        end_time = time.time()
        # 计算并输出推理时间
        inference_time = (end_time - start_time) * 1000  # 转换为毫秒
        # print(f"Inference time for {filename}: {inference_time:.2f} ms")
        print(filename)
        print(engine.get_binding_shape(1))

        seg_pred_trt = np.array(h_output_seg).reshape(engine.get_binding_shape(1))
        ver_pred_trt = np.array(h_output_ver).reshape(engine.get_binding_shape(2))
        print(seg_pred_trt[0, 0 ,205, 325])
        print(seg_pred_trt[0, 1 ,205, 325])
        # print(seg_pred_trt.shape)
        # print(ver_pred_trt.shape)
    print("done")

if __name__ == "__main__":

    engine_file = "models/model_demo.engine"  # TensorRT 引擎文件路径

    run_save2npy(engine_file)