//
//  operations.metal
//  metal-matmul
//
//  Created by Ankush Chatterjee on 10/11/24.
//

typedef struct MatrixDescriptor {
  int a_rows, a_cols;
  int b_rows, b_cols;
} MatrixDescriptor;

kernel void matmul(device const float* matA,
                   device const float* matB,
                   device float* result,
                   device const MatrixDescriptor* params,
                   uint2 index [[thread_position_in_grid]])
{
    if (index.y >= params->a_rows || index.x >= params->b_cols)
        return;
    int k = params->b_rows;
    int rIndex = index.y * params->b_cols + index.x;
    result[rIndex] = 0;
    // TOOD: do loop unrolling to improve perf
    for (int i=0; i < k; i++) {
        // result[index.y][index.x] += matA[index.y][i] * matB[i][index.x]
        result[rIndex] += matA[index.y * params->a_cols + i] * matB[i * params->b_cols + index.x];
    }
}
