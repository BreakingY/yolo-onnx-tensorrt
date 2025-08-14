# yolo-onnx-tensorrt
# Prepare
* pip install ultralytics
# Train
* python train.py
# ONNX
* python yolo_onnx_inference.py
# TensorRT
* /data/sunkx/TensorRT-8.5.1.7/bin/trtexec --onnx=runs/custom-yolov11s/weights/best.onnx --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:4x3x640x640 --saveEngine=best.engine --fp16
* g++ yolo_trt_inference.cpp -o yolo_trt_inference -I/data/sunkx/TensorRT-8.5.1.7/include -I/usr/local/cuda/include -I/usr/local/opencvgpu/include/opencv4 -L/data/sunkx/TensorRT-8.5.1.7/lib -L/usr/local/cuda/lib64 -L/usr/local/opencvgpu/lib -lnvinfer -lcudart -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -g -std=c++17
* ./yolo_trt_inference ./best.engine ./test.jpg