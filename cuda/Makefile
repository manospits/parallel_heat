CUDA_INSTALL_PATH = /opt/cuda
CC = g++
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
FLAGS = -I${CUDA_INSTALL_PATH}/include -Wall -g
NVFLAGS = -O2 -I${CUDA_INSTALL_PATH}/include --compiler-options -fno-strict-aliasing --ptxas-options=-v -g
BITS = $(shell getconf LONG_BIT)
ifeq (${BITS},64)
	LIBSUFFIX := 64
endif
LFLAGS = -L${CUDA_INSTALL_PATH}/lib${LIBSUFFIX} -lm -lstdc++ -lcudart
CLEAN_FILES = cuda_heat cuda_heat.o

cuda_heat: cuda_heat.o
	${CC} ${LFLAGS} -o $@ $^

cuda_heat.o: cuda_heat.cu timestamp.h
	${NVCC} ${NVFLAGS} -DUNIX -c cuda_heat.cu -o $@

clean:
	\rm -f $(CLEAN_FILES)

rebuild: clean test

