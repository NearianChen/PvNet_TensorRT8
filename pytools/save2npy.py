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

def imagePreprocess(image_path):
    from PIL import Image
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

    image_demo = np.array(Image.open(image_path)).astype(np.float32)
    inp = (((image_demo/255.)-mean)/std).transpose(2, 0, 1).astype(np.float32)
    return inp

def run_saveTRT2npy(engine_file_path, demo_images, output_dir_seg, output_dir_ver, mean, std):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    engine = load_engine(engine_file_path)
    context = engine.create_execution_context()
            
    h_input, h_output_seg, h_output_ver, d_input, d_output_seg, d_output_ver, stream = allocate_buffers(engine)
    for filename in os.listdir(demo_images):
        img_path = os.path.join(demo_images, filename)
        inp = imagePreprocess(img_path)
        inp_trt = inp[None, :]
        seg_pred_trt, ver_pred_trt = infer_tensorrt(
            context, h_input, h_output_seg, h_output_ver, 
            d_input, d_output_seg, d_output_ver, 
            stream, inp_trt
        )

        seg_pred_trt = np.array(h_output_seg).reshape(engine.get_binding_shape(1))
        ver_pred_trt = np.array(h_output_ver).reshape(engine.get_binding_shape(2))

        print(ver_pred_trt.shape)
        # 保存结果为 .npy 文件
        vertex_output_path = os.path.join(output_dir_ver, f"{filename}_vertex.npy")
        seg_output_path = os.path.join(output_dir_seg, f"{filename}_seg.npy")
        np.save(seg_output_path, seg_pred_trt)
        np.save(vertex_output_path, ver_pred_trt)
    print("done")

if __name__ == "__main__":

    engine_file = "models/model_demo.engine"  # TensorRT 引擎文件路径
    demo_images_dir = 'data/demo_cat'
    output_dir_seg = 'npy_files/trt_npy_v2/seg_npy'
    output_dir_ver = 'npy_files/trt_npy_v2/ver_npy'
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    run_saveTRT2npy(engine_file, demo_images_dir, output_dir_seg, output_dir_ver, mean, std)