#include <bits/stdc++.h>

using namespace std;


void readMatrix(const string &path, vector<double> &matrix, int rows, int cols) {
    FILE *fp = fopen(path.c_str(), "rb");
    if (!fp) {
        cerr << "Failed to open file: " << path << "\n";
        exit(1);
    }
    fread(matrix.data(), sizeof(double), rows * cols, fp);
    fclose(fp);
}

void writeMatrix(const string &path, const vector<double> &matrix, int rows, int cols) {
    FILE *fp = fopen(path.c_str(), "wb");
    if (!fp) {
        cerr << "Failed to open file: " << path << "\n";
        exit(1);
    }
    fwrite(matrix.data(), sizeof(double), rows * cols, fp);
    fclose(fp);
}

void matrixMultiplyIJK(const vector<double> &A, const vector<double> &B, vector<double> &C, int m, int n, int p) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

void matrixMultiplyIKJ(const vector<double> &A, const vector<double> &B, vector<double> &C, int m, int n, int p) {
    for (int i = 0; i < m; ++i) {
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < p; ++j) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

void matrixMultiplyJIK(const vector<double> &A, const vector<double> &B, vector<double> &C, int m, int n, int p) {
    for (int j = 0; j < p; ++j) {
        for (int i = 0; i < m; ++i) {
            for (int k = 0; k < n; ++k) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

void matrixMultiplyJKI(const vector<double> &A, const vector<double> &B, vector<double> &C, int m, int n, int p) {
    for (int j = 0; j < p; ++j) {
        for (int k = 0; k < n; ++k) {
            for (int i = 0; i < m; ++i) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

void matrixMultiplyKIJ(const vector<double> &A, const vector<double> &B, vector<double> &C, int m, int n, int p) {
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < p; ++j) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

void matrixMultiplyKJI(const vector<double> &A, const vector<double> &B, vector<double> &C, int m, int n, int p) {
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < p; ++j) {
            for (int i = 0; i < m; ++i) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}


int main(int argc, char **argv) {
    if (argc != 7) {
        cerr << "Usage: " << argv[0] << " <type> <mtx_A_rows> <mtx_A_cols> <mtx_B_cols> <input_path> <output_path>\n";
        return 1;
    }

    int type = atoi(argv[1]);
    int m = atoi(argv[2]); // Rows of A
    int n = atoi(argv[3]); // Columns of A (Rows of B)
    int p = atoi(argv[4]); // Columns of B
    string inputPath = argv[5];
    string outputPath = argv[6];

    // Read matrices A and B
    vector<double> A(m * n), B(n * p), C(m * p, 0);
    readMatrix(inputPath + "/mtx_A.bin", A, m, n);
    readMatrix(inputPath + "/mtx_B.bin", B, n, p);

    auto start = chrono::high_resolution_clock::now();
    switch (type) {
        case 0:
            matrixMultiplyIJK(A, B, C, m, n, p);
            break;
        case 1:
            matrixMultiplyIKJ(A, B, C, m, n, p);
            break;
        case 2:
            matrixMultiplyJIK(A, B, C, m, n, p);
            break;
        case 3:
            matrixMultiplyJKI(A, B, C, m, n, p);
            break;
        case 4:
            matrixMultiplyKIJ(A, B, C, m, n, p);
            break;
        case 5:
            matrixMultiplyKJI(A, B, C, m, n, p);
            break;
        default:
            cerr << "Invalid type\n";
            return 1;
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "Time elapsed: " << elapsed.count() << " seconds\n";

    // Write result matrix
    writeMatrix(outputPath + "/mtx_C.bin", C, m, p);

    return 0;
}
