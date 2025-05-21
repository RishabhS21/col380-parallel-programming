#include "modify.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

// Error checking macro for CUDA calls
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Kernel to compute histogram of matrix elements
__global__ void computeHistogramKernel(const int* input, int* histogram, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int val = input[idx];
        atomicAdd(&histogram[val], 1);
    }
}

// Kernel to build the prefix sum array
__global__ void prefixSumKernel(int* histogram, int* prefixSum, int size) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    
    // Load data into shared memory
    if (tid < size) {
        temp[tid] = histogram[tid];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();
    
    // Build prefix sum in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }
    
    // Write results to global memory
    if (tid < size) {
        prefixSum[tid] = temp[tid];
    }
}

// Kernel to rearrange matrix elements based on prefix sum
__global__ void rearrangeMatrixKernel(const int* input, int* output, int* positions, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int val = input[idx];
        int pos = atomicAdd(&positions[val], 1);
        output[pos] = val;
    }
}

// Process a single matrix
vector<vector<int>> processMatrix(const vector<vector<int>>& matrix, int range) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    int size = rows * cols;
    
    // Flatten the matrix for GPU processing
    vector<int> flatInput(size);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flatInput[i * cols + j] = matrix[i][j];
        }
    }
    
    // Allocate device memory
    int *d_input, *d_output, *d_histogram, *d_prefixSum, *d_positions;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_histogram, (range + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prefixSum, (range + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_positions, (range + 1) * sizeof(int)));
    
    // Copy data to device and initialize
    CUDA_CHECK(cudaMemcpy(d_input, flatInput.data(), size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_histogram, 0, (range + 1) * sizeof(int)));
    
    // Compute histogram
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    computeHistogramKernel<<<gridSize, blockSize>>>(d_input, d_histogram, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // For simplicity and reliability, we'll compute the prefix sum on the CPU
    vector<int> histogram(range + 1);
    CUDA_CHECK(cudaMemcpy(histogram.data(), d_histogram, (range + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    
    vector<int> prefixSum(range + 1);
    prefixSum[0] = 0;
    for (int i = 1; i <= range; i++) {
        prefixSum[i] = prefixSum[i-1] + histogram[i-1];
    }
    
    // Copy prefix sum to device
    CUDA_CHECK(cudaMemcpy(d_prefixSum, prefixSum.data(), (range + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_positions, d_prefixSum, (range + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
    
    // Rearrange matrix elements
    rearrangeMatrixKernel<<<gridSize, blockSize>>>(d_input, d_output, d_positions, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    vector<int> flatOutput(size);
    CUDA_CHECK(cudaMemcpy(flatOutput.data(), d_output, size * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_histogram));
    CUDA_CHECK(cudaFree(d_prefixSum));
    CUDA_CHECK(cudaFree(d_positions));
    
    // Reconstruct 2D matrix
    vector<vector<int>> result(rows, vector<int>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = flatOutput[i * cols + j];
        }
    }
    
    return result;
}

// Main function to process all matrices
vector<vector<vector<int>>> modify(vector<vector<vector<int>>>& matrices, vector<int>& ranges) {
    int numMatrices = matrices.size();
    vector<vector<vector<int>>> result(numMatrices);
    
    // Process multiple matrices
    for (int i = 0; i < numMatrices; i++) {
        result[i] = processMatrix(matrices[i], ranges[i]);
    }
    
    return result;
}
