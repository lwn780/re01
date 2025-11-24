#include <stdio.h>
#include <stdlib.h>

#define BDIMX 16
#define BDIMY 16

__global__ void transposeSmem(float *out, const float *in, int nx, int ny)
{
    __shared__ float tile[BDIMY][BDIMX];

    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int ti = iy * nx + ix;   

    // Compute thread index mapping inside block for transpose
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;  
    unsigned int icol = bidx % blockDim.y;   

    unsigned int ox = blockIdx.y * blockDim.y + icol;
    unsigned int oy = blockIdx.x * blockDim.x + irow;
    unsigned int to = oy * ny + ox;  // output index

    if (ix < nx && iy < ny) {
        tile[threadIdx.y][threadIdx.x] = in[ti];
    }

    __syncthreads();

    if (ox < ny && oy < nx) {
        out[to] = tile[icol][irow];
    }
}

int main()
{
    int nx = 10;   // matrix width
    int ny = 10;   // matrix height
    int nBytes = nx * ny * sizeof(float);

    printf("Matrix size: %d x %d\n", nx, ny);

    // allocate host
    float *h_in = (float *)malloc(nBytes);
    float *h_out = (float *)malloc(nBytes);

    // init input (A[y][x] = y * 10 + x)
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            h_in[y * nx + x] = y * nx + x;
        }
    }


    // allocate device
    float *d_in, *d_out;
    cudaMalloc(&d_in, nBytes);
    cudaMalloc(&d_out, nBytes);

    // copy input
    cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice);

    // launch kernel
    dim3 block(BDIMX, BDIMY);
    dim3 grid((nx + BDIMX - 1) / BDIMX,
              (ny + BDIMY - 1) / BDIMY);

    transposeSmem<<<grid, block>>>(d_out, d_in, nx, ny);

    cudaDeviceSynchronize();

    // copy result
    cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
