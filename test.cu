#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void transposeAddKernel(const float *A, const float *B, float *C, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 防止bank conflict

    int x = blockIdx.x * TILE_DIM + threadIdx.x; // col
    int y = blockIdx.y * TILE_DIM + threadIdx.y; // row

    // 从 A 读入 tile 并存到 shared memory 
    if (x < M && y < N)
        tile[threadIdx.y][threadIdx.x] = A[y * M + x];
    __syncthreads();

    // 写出转置 + 加法结果到 C 
    int trans_x = blockIdx.y * TILE_DIM + threadIdx.x; // 原来的 y -> 新的列
    int trans_y = blockIdx.x * TILE_DIM + threadIdx.y; // 原来的 x -> 新的行

    if (trans_x < N && trans_y < M) {
        float a_val = tile[threadIdx.x][threadIdx.y]; // 访问转置后的值
        float b_val = B[trans_y * N + trans_x];
        C[trans_y * N + trans_x] = a_val + b_val;
    }
}


int main() {
    // 输入矩阵 
    int N = 2; // A 的行数
    int M = 3; // A 的列数

    float A[6] = {1, 2, 3,
                  4, 5, 6};

    float B[6] = {10, 20,
                  30, 40,
                  50, 60};

    // C = A^T + B
    float C[6];

    // 设备内存分配 
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * M * sizeof(float));
    cudaMalloc(&d_C, N * M * sizeof(float));

    // 拷贝数据到 GPU 
    cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * M * sizeof(float), cudaMemcpyHostToDevice);

    //启动核函数 
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    transposeAddKernel<<<grid, block>>>(d_A, d_B, d_C, M, N);
    cudaDeviceSynchronize();

    // 拷贝结果回主机
    cudaMemcpy(C, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result C = A^T + B:\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%6.1f ", C[i * N + j]);
        }
        printf("\n");
    }


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
