# Makefile for yolo_trt_inference
CXX := g++
CXXFLAGS := -g -std=c++17 \
            -I/data/sunkx/TensorRT-10.4.0.26/include \
            -I/usr/local/cuda/include \
            -I/usr/local/include/opencv4 \
            -I./ByteTrack-cpp/include \
            -I/usr/include/eigen3/


LDFLAGS := -L/data/sunkx/TensorRT-10.4.0.26/lib \
           -L/usr/local/cuda/lib64 \
           -L/usr/local/lib


LDLIBS := -lnvinfer -lcudart \
          -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio \
		  -lnppig -lnppidei -lnppial


TARGET := yolo_trt_inference
SRC := yolo_trt_inference.cpp \
        ./ByteTrack-cpp/src/BYTETracker.cpp ./ByteTrack-cpp/src/KalmanFilter.cpp ./ByteTrack-cpp/src/lapjv.cpp ./ByteTrack-cpp/src/Object.cpp \
        ./ByteTrack-cpp/src/Rect.cpp ./ByteTrack-cpp/src/STrack.cpp


all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS) $(LDLIBS)


clean:
	rm -f $(TARGET)
