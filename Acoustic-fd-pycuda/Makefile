CUDA_INSTALL_PATH=/usr/local/cuda/
MPI_INSTALL_PATH=/usr

NVCC=$(CUDA_INSTALL_PATH)/bin/nvcc
MPICC=$(MPI_INSTALL_PATH)/bin/gcc

LDFLAGS=-L$(CUDA_INSTALL_PATH)/lib64
LIB=-lcudart -lcurand

CFILES=Toa_3dfd_mpi.c
CUFILES=Toa_3dfd_cuda.cu
OBJECTS=Toa_3dfd_mpi.o Toa_3dfd_cuda.o 
EXECNAME=fd

#CFILES = test.c
#CUFILES = test_cuda.cu
#OBJECTS = test.o test_cuda.o 
#EXECNAME = test

all:
	$(NVCC) ./src/Toa_3dfd_cuda.cu -shared -Xcompiler -fPIC -o ./lib/acoustic2order.so
	$(NVCC) ./src/Toa_gpu_3diso_fd_1orderfunciton.cu -shared -Xcompiler -fPIC -o ./lib/acoustic1order.so
	$(NVCC) ./src/Toa_gpu_3dvti_fd_1orderfunciton.cu -shared -Xcompiler -fPIC -o ./lib/acoustic1vti.so

clean:
	rm -f *.o $(EXECNAME)

