#include "template.hpp"
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <mpi.h>

using namespace std;

void init_mpi(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
}

void end_mpi() {
    MPI_Finalize();
}

// a helper function to distribute computation evenly
pair<int, int> get_partition(int rank, int size, int total) {
    int base = total / size;
    int remainder = total % size;
    int start = rank * base + min(rank, remainder);
    int count = base + (rank < remainder ? 1 : 0);
    return {start, count};
}

vector<vector<int>> degree_cen(vector<pair<int, int>>& partial_edge_list,
                               map<int, int>& partial_vertex_color,
                               int k) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Build local adjacency
    unordered_map<int, vector<int>> local_adjacency;
    for (const auto& edge : partial_edge_list) {
        local_adjacency[edge.first].push_back(edge.second);
        local_adjacency[edge.second].push_back(edge.first);
    }
    
    // Gather local color data
    vector<int> local_color_data;
    for (auto& vc : partial_vertex_color) {
        local_color_data.push_back(vc.first);
        local_color_data.push_back(vc.second);
    }
    
    int local_color_size = static_cast<int>(local_color_data.size());
    vector<int> color_sizes(size);
    MPI_Allgather(&local_color_size, 1, MPI_INT, color_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Prepare displacements
    vector<int> color_displs(size, 0);
    for (int i = 1; i < size; i++) {
        color_displs[i] = color_displs[i - 1] + color_sizes[i - 1];
    }
    
    int total_color_data_size = 0;
    for (int s : color_sizes) total_color_data_size += s;
    vector<int> all_color_data(total_color_data_size);
    
    // Gather all color data
    MPI_Allgatherv(local_color_data.data(), local_color_size, MPI_INT,
                   all_color_data.data(), color_sizes.data(),
                   color_displs.data(), MPI_INT, MPI_COMM_WORLD);
    
    map<int, int> global_vertex_color;
    set<int> unique_colors;
    
    for (int i = 0; i < total_color_data_size; i += 2) {
        int v = all_color_data[i];
        int c = all_color_data[i+1];
        global_vertex_color[v] = c;
        unique_colors.insert(c);
    }
    
    // Collect colors
    vector<int> colors(unique_colors.begin(), unique_colors.end());
    sort(colors.begin(), colors.end());
    
    // Prepare adjacency data for communication
    vector<int> adjacency_data;
    for (auto& kv : local_adjacency) {
        adjacency_data.push_back(kv.first);
        adjacency_data.push_back((int)kv.second.size());
        for (int neighbor : kv.second) {
            adjacency_data.push_back(neighbor);
        }
    }
    
    int local_adj_size = (int)adjacency_data.size();
    vector<int> adj_sizes(size);
    MPI_Allgather(&local_adj_size, 1, MPI_INT, adj_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    vector<int> adj_displs(size, 0);
    for (int i = 1; i < size; i++) {
        adj_displs[i] = adj_displs[i - 1] + adj_sizes[i - 1];
    }
    int total_adj_size = 0;
    for (int s : adj_sizes) total_adj_size += s;
    vector<int> all_adj_data(total_adj_size);
    
    // Gather adjacency
    MPI_Allgatherv(adjacency_data.data(), local_adj_size, MPI_INT,
                   all_adj_data.data(), adj_sizes.data(), adj_displs.data(),
                   MPI_INT, MPI_COMM_WORLD);
    
    unordered_map<int, vector<int>> global_adjacency;
    {
        int idx = 0;
        while (idx < total_adj_size) {
            int v = all_adj_data[idx++];
            int deg = all_adj_data[idx++];
            auto& nbrVec = global_adjacency[v];
            nbrVec.reserve(deg);
            for (int i = 0; i < deg; i++) {
                nbrVec.push_back(all_adj_data[idx++]);
            }
        }
    }
    
    // Build sorted list of vertices
    vector<int> all_vertices;
    all_vertices.reserve(global_vertex_color.size());
    for (auto& kv : global_vertex_color) {
        all_vertices.push_back(kv.first);
    }
    sort(all_vertices.begin(), all_vertices.end());
    
    // Partition among ranks
    auto partition_result = get_partition(rank, size, (int)all_vertices.size());
    int start_idx = partition_result.first;
    int vertex_count = partition_result.second;
    
    // Precompute color -> index for quick lookup
    unordered_map<int,int> color_to_idx;
    for (int i = 0; i < (int)colors.size(); i++) {
        color_to_idx[colors[i]] = i;
    }
    
    // Compute color-degree centrality for assigned vertices
    vector<int> centrality_data;
    centrality_data.reserve(vertex_count * colors.size() * 3);
    
    for (int i = 0; i < vertex_count; i++) {
        int vertex = all_vertices[start_idx + i];
        auto itAdj = global_adjacency.find(vertex);
        if (itAdj == global_adjacency.end()) continue;
        
        auto& neighbors = itAdj->second;
        vector<int> color_counts(colors.size(), 0);
        
        for (int neighbor : neighbors) {
            auto itColor = global_vertex_color.find(neighbor);
            if (itColor != global_vertex_color.end()) {
                int c = itColor->second;
                auto ctIt = color_to_idx.find(c);
                if (ctIt != color_to_idx.end()) {
                    color_counts[ctIt->second]++;
                }
            }
        }
        
        // Fill in local results
        for (int j = 0; j < (int)colors.size(); j++) {
            centrality_data.push_back(vertex);
            centrality_data.push_back(colors[j]);
            centrality_data.push_back(color_counts[j]);
        }
    }
    
    // Gather results from all ranks
    int local_centrality_size = (int)centrality_data.size();
    vector<int> centrality_sizes(size);
    MPI_Allgather(&local_centrality_size, 1, MPI_INT, centrality_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    vector<int> centrality_displs(size, 0);
    for (int i = 1; i < size; i++) {
        centrality_displs[i] = centrality_displs[i - 1] + centrality_sizes[i - 1];
    }
    int total_centrality_size = 0;
    for (int s : centrality_sizes) total_centrality_size += s;
    vector<int> all_centrality_data(total_centrality_size);
    
    MPI_Allgatherv(centrality_data.data(), local_centrality_size, MPI_INT,
                   all_centrality_data.data(), centrality_sizes.data(), centrality_displs.data(),
                   MPI_INT, MPI_COMM_WORLD);
    
    // Assemble final result on rank 0
    vector<vector<int>> result;
    if (rank == 0) {
        vector<vector<pair<int,int>>> color_buckets(colors.size());
        
        for (int i = 0; i < total_centrality_size; i += 3) {
            int v = all_centrality_data[i];
            int c = all_centrality_data[i+1];
            int score = all_centrality_data[i+2];
            
            auto ctIt = color_to_idx.find(c);
            if (ctIt != color_to_idx.end()) {
                color_buckets[ctIt->second].push_back({v, score});
            }
        }
        
        // Sort and store top k scores for each color
        // result.resize(colors.size());
        // for (size_t i = 0; i < colors.size(); i++) {
        //     auto& bucket = color_buckets[i];
        //     sort(bucket.begin(), bucket.end(), [](auto &a, auto &b){
        //         return (a.second > b.second) || ((a.second == b.second) && (a.first < b.first));
        //     });
        //     int limit = min(k, (int)bucket.size());
        //     result[i].reserve(limit);
        //     for (int j = 0; j < limit; j++) {
        //         result[i].push_back(bucket[j].first);
        //     }
        // }
        result.resize(colors.size());
        for (size_t i = 0; i < colors.size(); i++) {
            auto& bucket = color_buckets[i];
            
            int n = static_cast<int>(bucket.size());
            int limit = min(k, n);

            if (limit > 0 && limit < n) {
                nth_element(
                    bucket.begin(), 
                    bucket.begin() + limit, 
                    bucket.end(),
                    [](auto &a, auto &b) {
                        // Sort by descending score; tie-break on ascending vertex
                        if (a.second != b.second) return a.second > b.second;
                        return a.first < b.first;
                    }
                );
                // Discard everything except the top limit
                bucket.resize(limit);
            }

            // Sort just those top 'limit' elements in O(k log k)
            sort(
                bucket.begin(),
                bucket.end(),
                [](auto &a, auto &b) {
                    if (a.second != b.second) return a.second > b.second;
                    return a.first < b.first;
                }
            );

            // Reserve and copy out final top k
            result[i].reserve(bucket.size());
            for (auto &p : bucket) {
                result[i].push_back(p.first);
            }
        }
    }
    
    return result;
}