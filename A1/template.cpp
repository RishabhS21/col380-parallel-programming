#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
#include <set>
#include <omp.h>
#include "check.h"

using namespace std;

// Helper function to generate random non-zero blocks
vector<vector<int>> generate_random_block(int m) {
    vector<vector<int>> block(m, vector<int>(m));
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(0, 256);  // Range 1-256 to ensure non-zero
    
    bool has_nonzero = false;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            block[i][j] = dist(gen);
            if (block[i][j] != 0){
                has_nonzero = true;
            }
        }
    }
    // Ensure at least one non-zero element
    if (!has_nonzero) {
        uniform_int_distribution<int> ind_dist(0, m-1);
        block[ind_dist(gen)][ind_dist(gen)] = dist(gen) % 256 + 1;
    }
    return block;
}

map<pair<int, int>, vector<vector<int>>> generate_matrix(int n, int m, int b) {
    map<pair<int, int>, vector<vector<int>>> matrix_map;
    int num_blocks = n/m;

    // Generate b unique random positions
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(0, num_blocks - 1);
    set<pair<int, int>> used_positions;
    while (used_positions.size() < b) {
        int i = dist(gen);
        int j = dist(gen);
        used_positions.insert({i, j});
    }
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (const auto& pos : used_positions) {
                #pragma omp task shared(matrix_map) if(black_box())
                {
                    vector<vector<int>> block = generate_random_block(m);
                    #pragma omp critical
                    {
                        matrix_map[pos] = block;
                    }
                }
            }
        }
    }
    
    return matrix_map;
}

// Helper function for block multiplication
vector<vector<int>> multiply_blocks(const vector<vector<int>>& A, const vector<vector<int>>& B, int m, int block_row, vector<float>& row_statistics) {
    vector<vector<int>> result(m, vector<int>(m, 0));
    for (int k = 0; k < m; k += 4) {
        for (int i = 0; i < m; i++) {
            int global_row = block_row + i;
            int temp = 0, temp0 = 0, temp1 = 0, temp2 = 0, temp3 = 0;
            for (int j = 0; j < m; j++) {
                temp0 = A[i][k] * B[k][j];
                if (k + 1 < m) temp1 = A[i][k + 1] * B[k + 1][j];
                if (k + 2 < m) temp2 = A[i][k + 2] * B[k + 2][j];
                if (k + 3 < m) temp3 = A[i][k + 3] * B[k + 3][j];

                result[i][j] += (temp0 + temp1 + temp2 + temp3);
                if (temp0 != 0) temp++;
                if (temp1 != 0) temp++;
                if (temp2 != 0) temp++;
                if (temp3 != 0) temp++;

                #pragma omp atomic
                row_statistics[global_row] += temp;
            }
        }
    }
    return result;
}

vector<vector<int>> multiply_blocks2(const vector<vector<int>>& A, const vector<vector<int>>& B, int m) {
    vector<vector<int>> result(m, vector<int>(m, 0));
    for (int k = 0; k < m; k += 4) {
        for (int i = 0; i < m; i++) {
            // int global_row = block_row + i;
            for (int j = 0; j < m; j++) {
                result[i][j] += A[i][k] * B[k][j];
                if (k + 1 < m) result[i][j] += A[i][k + 1] * B[k + 1][j];
                if (k + 2 < m) result[i][j] += A[i][k + 2] * B[k + 2][j];
                if (k + 3 < m) result[i][j] += A[i][k + 3] * B[k + 3][j];
            }
        }
    }
    return result;
}
// Helper to check if a block is non-zero
bool is_non_zero(const vector<vector<int>>& block) {
    for (const auto& row : block) {
        for (int val : row) {
            if (val != 0) return true; // Found a nonzero element
        }
    }
    return false; // All elements are zero
}

// the required matrix multiplication function
vector<float> matmul(map<pair<int, int>, vector<vector<int>>>& blocks, int n, int m, int k) {
    vector<float> row_statistics(n, 0.0f);
    vector<int> row_elements(n, 0);  // Count of elements in non-zero blocks per row
    
    if (k == 0) {
        blocks.clear();
        for (int i = 0; i < n / m; i++) {
            blocks[{i, i}] = vector<vector<int>>(m, vector<int>(m, 0));
            for (int j = 0; j < m; j++) {
                blocks[{i, i}][j][j] = 1;
            }
        }
        return vector<float>();
    }

    // Pre-process: Replace multiples of 5 with 0
    for (auto it = blocks.begin(); it != blocks.end();) {
        bool has_nonzero = false;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                if (it->second[i][j] % 5 == 0) {
                    it->second[i][j] = 0;
                }
                if (it->second[i][j] != 0) has_nonzero = true;
            }
        }
        if (!has_nonzero) {
            it = blocks.erase(it);
        } else {
            ++it;
        }
    }
    
    // For k=2 case, we need to track statistics
    if (k == 2) {
        map<pair<int, int>, vector<vector<int>>> result;

        #pragma omp parallel
        {
            #pragma omp single
            {
                for (const auto& block1 : blocks) {
                    for (const auto& block2 : blocks) {
                        if (block1.first.second == block2.first.first) {
                            #pragma omp task shared(result, row_statistics) if(black_box())
                            {
                                int block_row = block1.first.first * m;
                                auto res_block = multiply_blocks(block1.second, block2.second, m, block_row, row_statistics);
                                pair<int, int> res_pos = {block1.first.first, block2.first.second};
                                
                                
                                #pragma omp critical
                                {
                                    if (!res_block.empty() and is_non_zero(res_block)) {
                                        if(result.find(res_pos) != result.end()) {
                                            auto& existing_block = result[res_pos];
                                            for (int i = 0; i < m; i++) {
                                                for (int j = 0; j < m; j++) {
                                                    existing_block[i][j] += res_block[i][j];
                                                }
                                            }
                                        } else {
                                            result[res_pos] = res_block;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        #pragma omp taskwait
        // Count elements in non-zero blocks including zero as well in the multiplied matrix
        for (const auto& block : result) {
            int block_row = block.first.first * m;
            for (int i = 0; i < m; i++) {
                int global_row = block_row + i;
                // for (int j = 0; j < m; j++) {
                    // if (block.second[i][j] != 0) {
                    row_elements[global_row]+=m;
                    // }
                // }
            }
        }
        // Normalize statistics
        for (int i = 0; i < n; i++) {
            if (row_elements[i] > 0) {
                row_statistics[i] /= row_elements[i];
            }
        }
        
        blocks = result;
        return row_statistics;
    }
    
    // For k > 2, perform matrix exponentiation
    else {
        map<pair<int, int>, vector<vector<int>>> current_result = blocks; // Start with A

        for (int power = 1; power < k; power++) {
            map<pair<int, int>, vector<vector<int>>> next_result;

            #pragma omp parallel
            {
                #pragma omp single
                {
                    for (const auto& block1 : current_result) {
                        for (const auto& block2 : blocks) {  // Always multiply with original matrix
                            if (block1.first.second == block2.first.first) {
                                #pragma omp task shared(next_result) if(black_box())
                                {
                                    auto res_block = multiply_blocks2(block1.second, block2.second, m);
                                    pair<int, int> res_pos = {block1.first.first, block2.first.second};
                                    
                                    // #pragma omp critical
                                    // {
                                    //     auto& existing_block = next_result[res_pos];
                                    //     for (int i = 0; i < m; i++) {
                                    //         for (int j = 0; j < m; j++) {
                                    //             existing_block[i][j] += res_block[i][j];
                                    //         }
                                    //     }
                                    // }
                                    #pragma omp critical
                                    {
                                        if (!res_block.empty() and is_non_zero(res_block)) {
                                            if(next_result.find(res_pos) != next_result.end()) {
                                                auto& existing_block = next_result[res_pos];
                                                for (int i = 0; i < m; i++) {
                                                    for (int j = 0; j < m; j++) {
                                                        existing_block[i][j] += res_block[i][j];
                                                    }
                                                }
                                            } else {
                                                next_result[res_pos] = res_block;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            #pragma omp taskwait
            current_result = next_result;
        }
        blocks = current_result; // Store final result
    }

    return vector<float>();
}