# yolo-onnx-tensorrt
# Prepare
* pip install ultralytics
# Train
* python train.py
# ONNX
* python yolo_onnx_inference.py
# TensorRT
* Test version: 8.5.1.7, 10.4.0.26
* TensorRT 10.4.0.26 loses more accuracy than 8.5.1.7, choose the version that best suits your needs.
* /data/sunkx/TensorRT-10.4.0.26/bin/trtexec --onnx=runs/custom-yolov11s/weights/best.onnx --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:4x3x640x640 --saveEngine=best.engine --fp16
* make
* ./yolo_trt_inference ./best.engine ./test.jpg