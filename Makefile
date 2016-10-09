INCLUDE= -I/home/ec2-user/pkg/nvidia_sdk_samples/deviceQuery/common/inc/
CFLAG= -std=c++11
.PHONY: all
all: matrix_gpu matrix_cpu

matrix_gpu: matrix_gpu.cu
	nvcc $(INCLUDE) $(CFLAG) -o matrix_gpu.exe matrix_gpu.cu

matrix_cpu: matrix_cpu.cu
	nvcc -o matrix_cpu.exe matrix_cpu.cu
.PHONY: clean
clean: 
	rm *.exe

