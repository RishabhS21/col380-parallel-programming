#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <limits>
#include <algorithm>
#include <utility>
#include <climits>
#include <queue>
#include <functional>
#include <sstream>
#include <chrono>

using namespace std;

typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pii;

struct pair_hash {
    inline size_t operator()(const pii& p) const {
        return p.first * 31 + p.second;
    }
};

struct SparseMatrix {
    int height;
    int width;
    int block_size;
    unordered_map<pii, vector<vector<ll>>, pair_hash> blocks;
    
    SparseMatrix() {}
    
    SparseMatrix(int h, int w, int bs) : height(h), width(w), block_size(bs) {}
    
    int nonZeroCount() const {
        int count = 0;
        for (const auto& entry : blocks) {
            for (const auto& row : entry.second) {
                for (const auto& val : row) {
                    if (val != 0) count++;
                }
            }
        }
        return count;
    }
    
    int nonZeroBlockCount() const {
        return blocks.size();
    }
    
    vector<char> serialize() const {
        size_t bufSize = 3 * sizeof(int) + sizeof(int);
        bufSize += blocks.size() * (2 * sizeof(int) + block_size * block_size * sizeof(ll));
        
        vector<char> buffer(bufSize);
        char* ptr = buffer.data();
        
        *((int*)ptr) = height; ptr += sizeof(int);
        *((int*)ptr) = width; ptr += sizeof(int);
        *((int*)ptr) = block_size; ptr += sizeof(int);
        *((int*)ptr) = blocks.size(); ptr += sizeof(int);
        
        for (const auto& entry : blocks) {
            *((int*)ptr) = entry.first.first; ptr += sizeof(int);
            *((int*)ptr) = entry.first.second; ptr += sizeof(int);
            
            for (const auto& row : entry.second) {
                for (const auto& val : row) {
                    *((ll*)ptr) = val; ptr += sizeof(ll);
                }
            }
        }
        
        return buffer;
    }
    
    static SparseMatrix deserialize(const vector<char>& buffer) {
        const char* ptr = buffer.data();
        
        int h = *((int*)ptr); ptr += sizeof(int);
        int w = *((int*)ptr); ptr += sizeof(int);
        int bs = *((int*)ptr); ptr += sizeof(int);
        int num_blocks = *((int*)ptr); ptr += sizeof(int);
        
        SparseMatrix matrix(h, w, bs);
        
        for (int i = 0; i < num_blocks; i++) {
            int row = *((int*)ptr); ptr += sizeof(int);
            int col = *((int*)ptr); ptr += sizeof(int);
            
            vector<vector<ll>> block(bs, vector<ll>(bs));
            for (int r = 0; r < bs; r++) {
                for (int c = 0; c < bs; c++) {
                    block[r][c] = *((ll*)ptr); ptr += sizeof(ll);
                }
            }
            
            matrix.blocks[{row, col}] = block;
        }
        
        return matrix;
    }
};

__global__ void blockMultiplyKernel(ll* A, ll* B, ll* C, int block_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < block_size && col < block_size) {
        unsigned long long sum = 0;
        
        __shared__ ll sharedA[32][32];
        __shared__ ll sharedB[32][32];
        
        for (int k = 0; k < block_size; k += blockDim.x) {
            if (k + threadIdx.x < block_size && row < block_size)
                sharedA[threadIdx.y][threadIdx.x] = A[row * block_size + (k + threadIdx.x)];
            else
                sharedA[threadIdx.y][threadIdx.x] = 0;
                
            if (k + threadIdx.y < block_size && col < block_size)
                sharedB[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * block_size + col];
            else
                sharedB[threadIdx.y][threadIdx.x] = 0;
                
            __syncthreads();
            
            #pragma unroll 16
            for (int j = 0; j < min(blockDim.x, block_size - k); j++) {
                unsigned long long a = (unsigned long long)sharedA[threadIdx.y][j];
                unsigned long long b = (unsigned long long)sharedB[j][threadIdx.x];
                sum += a * b;
            }
            
            __syncthreads();
        }
        
        C[row * block_size + col] = (ll)sum;
    }
}

__global__ void batchBlockMultiplyKernel(ll* batchA, ll* batchB, ll* batchC, 
                                       int block_size, int batch_size) {
    int block_id = blockIdx.z;
    
    if (block_id >= batch_size) return;
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < block_size && col < block_size) {
        int offset = block_id * block_size * block_size;
        ll* A = batchA + offset;
        ll* B = batchB + offset;
        ll* C = batchC + offset;
        
        unsigned long long sum = 0;
        
        __shared__ ll sharedA[32][32];
        __shared__ ll sharedB[32][32];
        
        for (int k = 0; k < block_size; k += 32) {
            if (k + threadIdx.x < block_size && row < block_size)
                sharedA[threadIdx.y][threadIdx.x] = A[row * block_size + k + threadIdx.x];
            else
                sharedA[threadIdx.y][threadIdx.x] = 0;
                
            if (k + threadIdx.y < block_size && col < block_size)
                sharedB[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * block_size + col];
            else
                sharedB[threadIdx.y][threadIdx.x] = 0;
                
            __syncthreads();
            
            #pragma unroll 8
            for (int j = 0; j < min(32, block_size - k); j++) {
                unsigned long long a = (unsigned long long)sharedA[threadIdx.y][j];
                unsigned long long b = (unsigned long long)sharedB[j][threadIdx.x];
                sum += a * b; 
            }
            
            __syncthreads();
        }
        
        C[row * block_size + col] = (ll)sum;
    }
}

double get_time() {
    return MPI_Wtime();
}

vector<vector<ll>> multiplyBlocksCPU(const vector<vector<ll>>& blockA, 
                                   const vector<vector<ll>>& blockB, 
                                   int block_size) {
    vector<vector<ll>> result(block_size, vector<ll>(block_size, 0));
    
    for (int i = 0; i < block_size; i++) {
        for (int k = 0; k < block_size; k++) {
            if (blockA[i][k] == 0) continue;
            unsigned long long val_A = (unsigned long long)blockA[i][k];
            
            for (int j = 0; j < block_size; j++) {
                if (blockB[k][j] == 0) continue;
                
                unsigned long long val_B = (unsigned long long)blockB[k][j];
                unsigned long long curr = (unsigned long long)result[i][j];
                result[i][j] = (ll)(curr + val_A * val_B); 
            }
        }
    }
    return result;
}

vector<vector<ll>> multiplyBlocksCUDA(const vector<vector<ll>>& blockA, 
                                    const vector<vector<ll>>& blockB, 
                                    int block_size,
                                    cudaStream_t& stream,
                                    ll* h_A, ll* h_B, ll* h_C,
                                    ll* d_A, ll* d_B, ll* d_C) {
    int size = block_size * block_size;
    
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            h_A[i * block_size + j] = blockA[i][j];
            h_B[i * block_size + j] = blockB[i][j];
        }
    }
    
    cudaMemcpyAsync(d_A, h_A, size * sizeof(ll), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, size * sizeof(ll), cudaMemcpyHostToDevice, stream);
    
    int threadsPerDim = min(32, block_size);
    dim3 threadsPerBlock(threadsPerDim, threadsPerDim);
    dim3 numBlocks((block_size + threadsPerDim - 1) / threadsPerDim, 
                   (block_size + threadsPerDim - 1) / threadsPerDim);
    
    blockMultiplyKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, block_size);
    
    cudaMemcpyAsync(h_C, d_C, size * sizeof(ll), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    vector<vector<ll>> result(block_size, vector<ll>(block_size));
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            result[i][j] = h_C[i * block_size + j];
        }
    }
    
    return result;
}

vector<vector<ll>> multiplyBlocksBatch(
    const vector<pair<vector<vector<ll>>, vector<vector<ll>>>>& blockPairs,
    int block_size, cudaStream_t& stream,
    ll* h_batchA, ll* h_batchB, ll* h_batchC,
    ll* d_batchA, ll* d_batchB, ll* d_batchC) {
    
    int batch_size = blockPairs.size();
    int single_size = block_size * block_size;
    int total_size = single_size * batch_size;
    
    #pragma omp parallel for if(batch_size > 4)
    for (int b = 0; b < batch_size; b++) {
        const vector<vector<ll>>& blockA = blockPairs[b].first;
        const vector<vector<ll>>& blockB = blockPairs[b].second;
        int offset = b * single_size;
        
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                h_batchA[offset + i * block_size + j] = blockA[i][j];
                h_batchB[offset + i * block_size + j] = blockB[i][j];
            }
        }
    }
    
    cudaMemcpyAsync(d_batchA, h_batchA, total_size * sizeof(ll), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_batchB, h_batchB, total_size * sizeof(ll), cudaMemcpyHostToDevice, stream);
    
    int threadsPerDim = min(32, block_size);
    dim3 threadsPerBlock(threadsPerDim, threadsPerDim);
    dim3 numBlocks((block_size + threadsPerDim - 1) / threadsPerDim, 
                  (block_size + threadsPerDim - 1) / threadsPerDim,
                  batch_size);
    
    batchBlockMultiplyKernel<<<numBlocks, threadsPerBlock, 0, stream>>>
        (d_batchA, d_batchB, d_batchC, block_size, batch_size);
    
    cudaMemcpyAsync(h_batchC, d_batchC, total_size * sizeof(ll), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    vector<vector<ll>> result(block_size, vector<ll>(block_size, 0));
    
    for (int b = 0; b < batch_size; b++) {
        int offset = b * single_size;
        
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                unsigned long long curr = (unsigned long long)result[i][j];
                unsigned long long batch_val = (unsigned long long)h_batchC[offset + i * block_size + j];
                result[i][j] = (ll)(curr + batch_val); 
            }
        }
    }
    
    return result;
}

SparseMatrix multiplyMatrices(const SparseMatrix& A, const SparseMatrix& B) {
    if (A.width != B.height) {
        cerr << "Matrix dimensions do not match: " << A.width << " != " << B.height << endl;
        exit(1);
    }
    
    SparseMatrix result(A.height, B.width, A.block_size);
    int block_size = A.block_size;
    int height_blocks = (A.height + block_size - 1) / block_size;
    int width_blocks = (B.width + block_size - 1) / block_size;
    int common_blocks = (A.width + block_size - 1) / block_size;
    
    const int numStreams = 8;  
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    int size = block_size * block_size;
    vector<ll*> h_A_arr(numStreams), h_B_arr(numStreams), h_C_arr(numStreams);
    vector<ll*> d_A_arr(numStreams), d_B_arr(numStreams), d_C_arr(numStreams);
    
    const int MAX_BATCH_SIZE = 64;
    vector<ll*> h_batchA_arr(numStreams), h_batchB_arr(numStreams), h_batchC_arr(numStreams);
    vector<ll*> d_batchA_arr(numStreams), d_batchB_arr(numStreams), d_batchC_arr(numStreams);
    
    for (int i = 0; i < numStreams; i++) {
        cudaMallocHost(&h_A_arr[i], size * sizeof(ll));
        cudaMallocHost(&h_B_arr[i], size * sizeof(ll));
        cudaMallocHost(&h_C_arr[i], size * sizeof(ll));
        
        cudaMalloc(&d_A_arr[i], size * sizeof(ll));
        cudaMalloc(&d_B_arr[i], size * sizeof(ll));
        cudaMalloc(&d_C_arr[i], size * sizeof(ll));
        
        cudaMallocHost(&h_batchA_arr[i], MAX_BATCH_SIZE * size * sizeof(ll));
        cudaMallocHost(&h_batchB_arr[i], MAX_BATCH_SIZE * size * sizeof(ll));
        cudaMallocHost(&h_batchC_arr[i], MAX_BATCH_SIZE * size * sizeof(ll));
        
        cudaMalloc(&d_batchA_arr[i], MAX_BATCH_SIZE * size * sizeof(ll));
        cudaMalloc(&d_batchB_arr[i], MAX_BATCH_SIZE * size * sizeof(ll));
        cudaMalloc(&d_batchC_arr[i], MAX_BATCH_SIZE * size * sizeof(ll));
    }
    
    vector<vector<int>> connections_A(height_blocks);
    vector<vector<int>> connections_B(common_blocks);
    
    for (const auto& entry : A.blocks) {
        connections_A[entry.first.first].push_back(entry.first.second);
    }
    
    for (const auto& entry : B.blocks) {
        connections_B[entry.first.first].push_back(entry.first.second);
    }
    
    vector<vector<bool>> potential_result(height_blocks, vector<bool>(width_blocks, false));
    
    for (int i = 0; i < height_blocks; i++) {
        for (int k : connections_A[i]) {
            for (int j : connections_B[k]) {
                potential_result[i][j] = true;
            }
        }
    }
    
    #pragma omp parallel
    {
        unordered_map<pii, vector<vector<ll>>, pair_hash> local_blocks;
        int thread_id = omp_get_thread_num() % numStreams;
        
        #pragma omp for schedule(dynamic, 32)  
        for (int i = 0; i < height_blocks; i++) {
            for (int j = 0; j < width_blocks; j++) {
                if (!potential_result[i][j]) continue;
                
                vector<vector<ll>> resultBlock(block_size, vector<ll>(block_size, 0));
                bool hasNonZero = false;
                
                vector<pair<vector<vector<ll>>, vector<vector<ll>>>> blockPairs;
                for (int k : connections_A[i]) {
                    auto itA = A.blocks.find({i, k});
                    if (itA == A.blocks.end()) continue;
                    
                    for (int col : connections_B[k]) {
                        if (col == j) {
                            auto itB = B.blocks.find({k, j});
                            if (itB != B.blocks.end()) {
                                blockPairs.push_back({itA->second, itB->second});
                            }
                        }
                    }
                }
                
                if (blockPairs.empty()) continue;

                if (block_size <= 8) {
                    for (const auto& pair : blockPairs) {
                        vector<vector<ll>> product = multiplyBlocksCPU(pair.first, pair.second, block_size);
                        
                        for (int r = 0; r < block_size; r++) {
                            for (int c = 0; c < block_size; c++) {
                                unsigned long long curr = (unsigned long long)resultBlock[r][c];
                                unsigned long long prod_val = (unsigned long long)product[r][c];
                                resultBlock[r][c] = (ll)(curr + prod_val); 
                                
                                if (resultBlock[r][c] != 0) hasNonZero = true;
                            }
                        }
                    }
                } else if (blockPairs.size() > 1 && blockPairs.size() <= MAX_BATCH_SIZE) {
                    vector<vector<ll>> product = multiplyBlocksBatch(
                        blockPairs, block_size, streams[thread_id],
                        h_batchA_arr[thread_id], h_batchB_arr[thread_id], h_batchC_arr[thread_id],
                        d_batchA_arr[thread_id], d_batchB_arr[thread_id], d_batchC_arr[thread_id]);
                    
                    for (int r = 0; r < block_size; r++) {
                        for (int c = 0; c < block_size; c++) {
                            resultBlock[r][c] = product[r][c];
                            if (resultBlock[r][c] != 0) hasNonZero = true;
                        }
                    }
                } else {
                    for (const auto& pair : blockPairs) {
                        vector<vector<ll>> product = multiplyBlocksCUDA(
                            pair.first, pair.second, block_size, streams[thread_id],
                            h_A_arr[thread_id], h_B_arr[thread_id], h_C_arr[thread_id],
                            d_A_arr[thread_id], d_B_arr[thread_id], d_C_arr[thread_id]);
                        
                        for (int r = 0; r < block_size; r++) {
                            for (int c = 0; c < block_size; c++) {
                                unsigned long long curr = (unsigned long long)resultBlock[r][c];
                                unsigned long long prod_val = (unsigned long long)product[r][c];
                                resultBlock[r][c] = (ll)(curr + prod_val); 
                                
                                if (resultBlock[r][c] != 0) hasNonZero = true;
                            }
                        }
                    }
                }
                
                if (hasNonZero) {
                    local_blocks[{i, j}] = move(resultBlock);
                }
            }
        }
        
        #pragma omp critical
        {
            for (auto& entry : local_blocks) {
                result.blocks[entry.first] = move(entry.second);
            }
        }
    }
    
    for (int i = 0; i < numStreams; i++) {
        cudaFreeHost(h_A_arr[i]);
        cudaFreeHost(h_B_arr[i]);
        cudaFreeHost(h_C_arr[i]);
        
        cudaFree(d_A_arr[i]);
        cudaFree(d_B_arr[i]);
        cudaFree(d_C_arr[i]);
        
        cudaFreeHost(h_batchA_arr[i]);
        cudaFreeHost(h_batchB_arr[i]);
        cudaFreeHost(h_batchC_arr[i]);
        
        cudaFree(d_batchA_arr[i]);
        cudaFree(d_batchB_arr[i]);
        cudaFree(d_batchC_arr[i]);
        
        cudaStreamDestroy(streams[i]);
    }
    
    return result;
}

SparseMatrix readMatrix(const string& filename, int block_size) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Cannot open file: " << filename << endl;
        exit(1);
    }
    
    int height, width;
    file >> height >> width;
    
    SparseMatrix matrix(height, width, block_size);
    
    int num_nonzero_blocks;
    file >> num_nonzero_blocks;
    
    matrix.blocks.reserve(num_nonzero_blocks);
    
    for (int i = 0; i < num_nonzero_blocks; i++) {
        int abs_row, abs_col;
        file >> abs_row >> abs_col;
        
        int block_row = abs_row / block_size;
        int block_col = abs_col / block_size;
        
        vector<vector<ll>> block(block_size, vector<ll>(block_size, 0));
        
        for (int r = 0; r < block_size; r++) {
            for (int c = 0; c < block_size; c++) {
                file >> block[r][c];
            }
        }
        
        matrix.blocks[{block_row, block_col}] = block;
    }
    
    file.close();
    return matrix;
}

void writeMatrix(const SparseMatrix& matrix, const string& filename) {
    ofstream file(filename, ios::binary);
    if (!file) {
        cerr << "Cannot open file for writing: " << filename << endl;
        exit(1);
    }
    
    file << matrix.height << " " << matrix.width << endl;
    file << matrix.blocks.size() << endl;
    
    file.rdbuf()->pubsetbuf(new char[32768], 32768);
    
    for (const auto& entry : matrix.blocks) {
        int abs_row = entry.first.first * matrix.block_size;
        int abs_col = entry.first.second * matrix.block_size;
        
        file << abs_row << " " << abs_col << endl;
        
        const vector<vector<ll>>& block = entry.second;
        for (int r = 0; r < matrix.block_size; r++) {
            for (int c = 0; c < matrix.block_size; c++) {
                file << block[r][c];
                if (c < matrix.block_size - 1) {
                    file << " ";
                }
            }
            file << endl;
        }
    }
    
    file.close();
}

double estimateCost(const SparseMatrix& A, const SparseMatrix& B) {
    int count = 0;
    int height_blocks = (A.height + A.block_size - 1) / A.block_size;
    int width_blocks = (B.width + B.block_size - 1) / B.block_size;
    int common_blocks = (A.width + A.block_size - 1) / A.block_size;
    
    vector<vector<int>> A_blocks(height_blocks);
    vector<vector<int>> B_cols(common_blocks);
    
    for (const auto& entry : A.blocks) {
        A_blocks[entry.first.first].push_back(entry.first.second);
    }
    
    for (const auto& entry : B.blocks) {
        B_cols[entry.first.first].push_back(entry.first.second);
    }
    
    for (int i = 0; i < height_blocks; i++) {
        if (A_blocks[i].empty()) continue;
        
        for (int j = 0; j < width_blocks; j++) {
            bool has_path = false;
            
            for (int k : A_blocks[i]) {
                if (k >= common_blocks) continue;
                
                for (int col : B_cols[k]) {
                    if (col == j) {
                        has_path = true;
                        count++;
                        break;
                    }
                }
                if (has_path) break;
            }
        }
    }
    
    return count * 1.0 / (height_blocks * width_blocks) * (A.height/64) * (B.width/64);
}

vector<int> findOptimalOrder(const vector<SparseMatrix>& matrices, int mpi_rank, int mpi_size) {
    int N = matrices.size();
    vector<int> order;
    
    if (N <= 2) {
        if (N == 2 && matrices[0].width != matrices[1].height) {
            if (matrices[1].width == matrices[0].height)
                return {1, 0};
        }
        order.resize(N);
        for (int i = 0; i < N; i++) order[i] = i;
        return order;
    }
    
    vector<vector<pair<int, double>>> local_graph(N);
    
    for (int i = 0; i < N; i++) {
        if (i % mpi_size != mpi_rank) continue;
        
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            
            if (matrices[i].width == matrices[j].height) {
                double cost = estimateCost(matrices[i], matrices[j]);
                local_graph[i].push_back({j, cost});
            }
        }
    }
    
    if (mpi_size > 1) {
        if (mpi_rank == 0) {
            for (int p = 1; p < mpi_size; p++) {
                for (int i = 0; i < N; i++) {
                    if (i % mpi_size != p) continue;
                    
                    int edges_count;
                    MPI_Recv(&edges_count, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    for (int e = 0; e < edges_count; e++) {
                        int target, cost_int;
                        MPI_Recv(&target, 1, MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Recv(&cost_int, 1, MPI_INT, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        double cost = static_cast<double>(cost_int) / 1000.0;
                        
                        local_graph[i].push_back({target, cost});
                    }
                }
            }
        } else {
            for (int i = 0; i < N; i++) {
                if (i % mpi_size != mpi_rank) continue;
                
                int edges_count = local_graph[i].size();
                MPI_Send(&edges_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                
                for (const auto& edge : local_graph[i]) {
                    int target = edge.first;
                    int cost_int = static_cast<int>(edge.second * 1000);
                    
                    MPI_Send(&target, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
                    MPI_Send(&cost_int, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
                }
            }
        }
    }
    
    if (mpi_rank == 0) {
        vector<bool> used(N, false);
        
        int start = 0;
        int max_connections = 0;
        
        for (int i = 0; i < N; i++) {
            if (local_graph[i].size() > max_connections) {
                max_connections = local_graph[i].size();
                start = i;
            }
        }
        
        order.push_back(start);
        used[start] = true;
        
        while (order.size() < N) {
            int last = order.back();
            int next = -1;
            double best_cost = numeric_limits<double>::max();
            
            for (const auto& edge : local_graph[last]) {
                int to = edge.first;
                double cost = edge.second;
                
                if (!used[to] && cost < best_cost) {
                    next = to;
                    best_cost = cost;
                }
            }
            
            if (next != -1) {
                order.push_back(next);
                used[next] = true;
            } else {
                for (int i = 0; i < N; i++) {
                    if (!used[i] && matrices[last].width == matrices[i].height) {
                        next = i;
                        break;
                    }
                }
                
                if (next != -1) {
                    order.push_back(next);
                    used[next] = true;
                } else {
                    for (int i = 0; i < N; i++) {
                        if (!used[i]) {
                            order.push_back(i);
                            used[i] = true;
                            break;
                        }
                    }
                }
            }
        }
        
        bool valid = true;
        for (int i = 0; i < N-1; i++) {
            int idx1 = order[i];
            int idx2 = order[i+1];
            if (matrices[idx1].width != matrices[idx2].height) {
                valid = false;
                break;
            }
        }
        
        if (!valid) {
            order.clear();
            fill(used.begin(), used.end(), false);
            
            order.push_back(0);
            used[0] = true;
            
            while (order.size() < N) {
                int last = order.back();
                bool found = false;
                
                for (int i = 0; i < N; i++) {
                    if (!used[i] && matrices[last].width == matrices[i].height) {
                        order.push_back(i);
                        used[i] = true;
                        found = true;
                        break;
                    }
                }
                
                if (!found) {
                    for (int i = 0; i < N; i++) {
                        if (!used[i]) {
                            order.push_back(i);
                            used[i] = true;
                            break;
                        }
                    }
                }
            }
            
            valid = true;
            for (int i = 0; i < N-1; i++) {
                int idx1 = order[i];
                int idx2 = order[i+1];
                if (matrices[idx1].width != matrices[idx2].height) {
                    valid = false;
                    break;
                }
            }
            
            if (!valid) {
                order.resize(N);
                for (int i = 0; i < N; i++) order[i] = i;
            }
        }
    }
    
    int order_size = N;
    if (mpi_rank == 0) {
        order_size = order.size();
    }
    
    MPI_Bcast(&order_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (mpi_rank != 0) {
        order.resize(order_size);
    }
    
    MPI_Bcast(order.data(), order_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    return order;
}

void distributeMatrixWork(const vector<SparseMatrix>& matrices, 
    const vector<int>& order,
    int mpi_rank, int mpi_size) {
    int N = matrices.size();

    vector<MPI_Request> requests;
    
    if (mpi_rank == 0) {
        SparseMatrix result = matrices[order[0]];
        int next_matrix = 1;

        while (next_matrix < N) {
            int matrix_idx = order[next_matrix];

            if (result.width == matrices[matrix_idx].height) {
                if (mpi_size == 1) {
                    result = multiplyMatrices(result, matrices[matrix_idx]);
                } 
                else {
                    int target_rank = 1 + (next_matrix - 1) % (mpi_size - 1);
                    
                    vector<char> buffer = result.serialize();
                    int msg_size = buffer.size();
                    
                    MPI_Request req;
                    MPI_Isend(&msg_size, 1, MPI_INT, target_rank, 0, MPI_COMM_WORLD, &req);
                    requests.push_back(req);
                    
                    MPI_Isend(buffer.data(), msg_size, MPI_CHAR, target_rank, 1, MPI_COMM_WORLD, &req);
                    requests.push_back(req);
                    
                    MPI_Isend(&matrix_idx, 1, MPI_INT, target_rank, 2, MPI_COMM_WORLD, &req);
                    requests.push_back(req);
                    
                    MPI_Status status;
                    int result_size;
                    
                    MPI_Recv(&result_size, 1, MPI_INT, target_rank, 3, MPI_COMM_WORLD, &status);
                    
                    vector<char> result_buffer(result_size);
                    MPI_Recv(result_buffer.data(), result_size, MPI_CHAR, target_rank, 4, MPI_COMM_WORLD, &status);
                    
                    result = SparseMatrix::deserialize(result_buffer);
                }
            }
            else {
                cerr << "Cannot multiply matrices: " << result.width << " != " << matrices[matrix_idx].height << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            next_matrix++;
        }

        if (!requests.empty()) {
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        }

        writeMatrix(result, "matrix");

        for (int i = 1; i < mpi_size; i++) {
            int terminate = -1;
            MPI_Request req;
            MPI_Isend(&terminate, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &req);
            MPI_Request_free(&req);  
        }
    }
    else {
        while (true) {
            MPI_Status status;
            int msg_size;

            MPI_Recv(&msg_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

            if (msg_size == -1) {
                break;
            }

            vector<char> buffer(msg_size);
            MPI_Recv(buffer.data(), msg_size, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &status);

            int matrix_idx;
            MPI_Recv(&matrix_idx, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);

            SparseMatrix A = SparseMatrix::deserialize(buffer);
            SparseMatrix result;

            if (A.width == matrices[matrix_idx].height) {
                result = multiplyMatrices(A, matrices[matrix_idx]);
            } else if (matrices[matrix_idx].width == A.height) {
                result = multiplyMatrices(matrices[matrix_idx], A);
            } else {
                cerr << "Cannot multiply matrices: " << A.width << " != " << matrices[matrix_idx].height << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            vector<char> result_buffer = result.serialize();
            int result_size = result_buffer.size();

            MPI_Send(&result_size, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
            MPI_Send(result_buffer.data(), result_size, MPI_CHAR, 0, 4, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    double s = get_time();

    if (argc != 2) {
        if (mpi_rank == 0) {
            cerr << "Usage: " << argv[0] << " <folder_path>" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    string folder_path = argv[1];
    string size_file = folder_path + "/size";
    
    ifstream size_file_stream(size_file);
    if (!size_file_stream) {
        if (mpi_rank == 0) {
            cerr << "Cannot open size file: " << size_file << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    int N, k;
    size_file_stream >> N >> k;
    size_file_stream.close();
    
    if (mpi_rank == 0) {
        cout << "Found " << N << " matrices with block size " << k << endl;
    }
    
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    cudaSetDevice(mpi_rank % 8);
    
    vector<SparseMatrix> matrices(N);
    
    double read_start = get_time();
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; i++) {
        if (i % mpi_size == mpi_rank) {
            string matrix_file = folder_path + "/matrix" + to_string(i+1);
            matrices[i] = readMatrix(matrix_file, k);
        }
    }
    
    for (int i = 0; i < N; i++) {
        int src_rank = i % mpi_size;
        if (mpi_rank == src_rank) {
            vector<char> buffer = matrices[i].serialize();
            int buffer_size = buffer.size();
            
            MPI_Bcast(&buffer_size, 1, MPI_INT, src_rank, MPI_COMM_WORLD);
            MPI_Bcast(buffer.data(), buffer_size, MPI_CHAR, src_rank, MPI_COMM_WORLD);
        } else {
            int buffer_size;
            MPI_Bcast(&buffer_size, 1, MPI_INT, src_rank, MPI_COMM_WORLD);
            
            vector<char> buffer(buffer_size);
            MPI_Bcast(buffer.data(), buffer_size, MPI_CHAR, src_rank, MPI_COMM_WORLD);
            
            matrices[i] = SparseMatrix::deserialize(buffer);
        }
    }
    double read_end = get_time();
    
    if (mpi_rank == 0) {
        cout << "Read all matrices in " << (read_end - read_start) << " seconds" << endl;
    }
    
    vector<int> order = findOptimalOrder(matrices, mpi_rank, mpi_size);
    
    if (mpi_rank == 0) {
        cout << "Multiplication order: ";
        for (int i = 0; i < min(10, (int)order.size()); i++) {
            cout << order[i] << " ";
        }
        if (order.size() > 10) cout << "...";
        cout << endl;
    }
    
    distributeMatrixWork(matrices, order, mpi_rank, mpi_size);
    
    double e = get_time();

    if(mpi_rank == 0){
        cout << "Total execution time: " << (e - s) << " seconds" << endl;
    }
    
    MPI_Finalize();
    return 0;
}