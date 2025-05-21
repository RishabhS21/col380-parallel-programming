MPICXX = mpicxx
CXX = g++
CXXFLAGS = -std=c++11 -O3 -fopenmp
CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
NVCCFLAGS = -O3 -Xcompiler -fopenmp
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lm

SOURCES = a4.cu

EXECUTABLE = a4

all: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS) -ccbin $(MPICXX)

clean:
	rm -f $(EXECUTABLE)