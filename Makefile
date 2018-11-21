CUDA_INSTALL_PATH=/usr/local/cuda/
MPI_INSTALL_PATH=/usr

NVCC=$(CUDA_INSTALL_PATH)/bin/nvcc

LDFLAGS=-L$(CUDA_INSTALL_PATH)/lib64
LIB=-lcudart -lcurand

all:
	$(NVCC) ./src/Toa_3dfd_cuda.cu -shared -Xcompiler -fPIC -o ./lib/acoustic2order.so
	$(NVCC) ./src/Toa_gpu_3diso_fd_1orderfunciton.cu -shared -Xcompiler -fPIC -o ./lib/acoustic1order.so
	$(NVCC) ./src/Toa_gpu_3dvti_fd_1orderfunciton.cu -shared -Xcompiler -fPIC -o ./lib/acoustic1vti.so

clean:
	rm -f ./lib/*.so

