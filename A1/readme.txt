1. Optimized Random Block Generation

Used random number generator to improve efficiency and avoid contention in multi-threaded environments.

2. Efficient Matrix Generation Using Tasks

Used OpenMP tasks to generate blocks in parallel while ensuring unique positions.

#pragma omp critical is used only when inserting into the shared matrix_map, reducing contention.

3. Loop Unrolling in Matrix Multiplication

Inner loop unrolled by a factor of 4 for better cache utilization and reduced loop overhead.

Local accumulation of multiplications before storing results.

4. Reduced Synchronization Overhead

Local accumulation of non-zero contributions per row before performing atomic updates.

Critical sections are minimized by only updating shared structures once per task completion.

5. Preprocessing for Efficient Computation

Multiples of 5 are set to zero before computation to avoid unnecessary multiplications.

Blocks that become entirely zero are removed, reducing the number of computations.

6. Task Grouping for Reduced Task Overhead

Blocks are grouped by their block row to minimize task creation overhead.

This improves locality and reduces redundant task scheduling.

7. Efficient Matrix Exponentiation

Matrix exponentiation for k > 2 is implemented using progressive multiplication.

Instead of squaring intermediate results, each step multiplies the previous result with the original matrix.

8. Thread-Safe Updates to Global Statistics

Used #pragma omp atomic for safe updates to row_statistics.

Ensured statistics normalization after task execution.

These optimizations together improve performance while maintaining correctness and compliance with OpenMP constraints.