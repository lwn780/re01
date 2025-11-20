#include <curand_kernel.h>
#include <iostream>
#include <cstdint>
#define SM 32
#define CUDA_CHECK_MSG(call,msg) \
do { \
    cudaError_t err=call;\
    if(err!=cudaSuccess){ \
        fprintf(stderr,"[%s] CUDA error at %s:%d:%s\n",msg,__FILE__,__LINE__,cudaGetErrorString(err));\
        exit(-1);\
    }\
}while(0);
using namespace std;
__global__ void generate_random(float *array, int n, unsigned seed) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  idx=idx*4;
  float4 tmp;
  curandState state;
  curand_init(seed, idx, 0, &state);
  if (idx+4 < n) {
    
    tmp.x = curand_uniform(&state);
    tmp.y=curand_uniform(&state);
    tmp.z=curand_uniform(&state);
    tmp.w=curand_uniform(&state);
    reinterpret_cast<float4 *>(&(array[idx]))[0]=tmp;
  }else{
    for(int i=0;idx+i<n;i++)
    {
      array[idx+i]=curand_uniform(&state);
    }
  }
}
__global__ void memcpy_1d_holed(const float *input,float *output,size_t input_size)
{
    int tid=(threadIdx.x+blockDim.x*blockIdx.x)*4;
    for(int i=0;i<4;i++)
    {
        if(tid+i<input_size)
        {
            float temp=input[tid+i];
            output[tid+i]=temp;
        }
    }
}
__global__ void memcpy_1d_vector(const float *input,float *output,size_t input_size)
{
    int tid=(threadIdx.x+blockDim.x*blockIdx.x)*4;
    bool aligned_input = ((uintptr_t) input % 16 == 0);
    bool aligned_output = ((uintptr_t) output % 16 == 0);
    float temp[4];
    if(aligned_input&&tid+4<=input_size)
    {
        reinterpret_cast<float4 *>(temp)[0]=reinterpret_cast<const float4 *>(&input[tid])[0];
    } else{
    for(int i=0;i<4;i++)
    {
        if(tid+i<input_size)
        {
            temp[i]=input[tid+i];
        }
    }
    }
    if(aligned_output&&tid+4<=input_size)
    {
        reinterpret_cast<float4 *>(&output[tid])[0]=reinterpret_cast<float4 *>(temp)[0];
    } else{
    for(int i=0;i<4;i++)
    {
        if(tid+i<input_size)
        {
            output[tid+i]=temp[i];
        }
    }
    }
}
__global__ void memcpy_1d_holed_another(const float *input,float *output,size_t input_size)
{
    int tid=(threadIdx.x+blockDim.x*blockIdx.x)*4;
    float temp[4];
    for(int i=0;i<4;i++)
    {
        if(tid+i<input_size)
        {
            temp[i]=input[tid+i];
        }
    }
    for(int i=0;i<4;i++)
    {
        if(tid+i<input_size)
        {
            output[tid+i]=temp[i];
        }
    }
}
__global__ void memcpy_1d_lyl(const float *input,float *output,size_t input_size)
{
    int tid=(threadIdx.x+blockDim.x*blockIdx.x)*4;
    float value0,value1,value2,value3;
    if(tid<input_size)
        value0 = input[tid+0];
    if(tid+1<input_size)
        value1= input[tid+1];
    if(tid+2<input_size)
        value2= input[tid+2];
    if(tid+3<input_size)
        value3= input[tid+3];
    if(tid<input_size)
        output[tid+0]=value0;
    if(tid+1<input_size)
        output[tid+1]=value1;
    if(tid+2<input_size)
        output[tid+2]=value2;
    if(tid+3<input_size)
        output[tid+3]=value3;
}
__global__ void memcpy_1d_diffsuse(const float *input,float *output,size_t input_size)
{
    int bid=blockIdx.x*4;
    int tid=threadIdx.x;
    float temp[4];
    for(int i=0;i<4;i++)
    {
        int id=(bid+i)*blockDim.x+tid;
        if(id<input_size)
        {
            temp[i]=input[id];
        }
    }
    for(int i=0;i<4;i++)
    {
        int  id=(bid+i)*blockDim.x+tid;
        if(id<input_size)
        {
            output[id]=temp[i];
        }
    }

}
__global__ void memcpy_2d_holed(const float *input,float *output,size_t row,size_t col)
{
    int xid= 4*(threadIdx.x+blockDim.x*blockDim.x);
    int yid= 4*(threadIdx.y+blockDim.y*blockDim.y);
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
        {
            if((yid+i)<row&&(xid+j)<col)
            {
                float temp=input[(yid+i)*col+xid+j];
                output[(yid+i)*col+xid+j]=temp;
            }
        }
    }
}
__global__ void memcpy_2d_holed_smem(const float *input,float *output,size_t row,size_t col)
{
    __shared__ float smem[SM*SM];
    int txid=4*threadIdx.x;
    int tyid=4*threadIdx.y;
    int gxid= 4*(threadIdx.x+blockDim.x*blockDim.x);
    int gyid= 4*(threadIdx.y+blockDim.y*blockDim.y);
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
        {
            if((gxid+j)<col&&(gyid+i)<row)
            {
                smem[(tyid+i)*SM+(txid+j)]=input[(gyid+i)*col+gxid+j];
            }
        }
    }
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
        {
            if((gxid+j)<col&&(gyid+i)<row)
            {
                output[(gyid+i)*col+gxid+j]=smem[(tyid+i)*SM+(txid+j)];
            }
        }
    }
}
__global__ void memcpy_2d_holed_reg(const float *input,float *output,size_t row,size_t col)
{
    float reg[4][4];
    int gxid= 4*(threadIdx.x+blockDim.x*blockDim.x);
    int gyid= 4*(threadIdx.y+blockDim.y*blockDim.y);
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
        {
            if((gxid+j)<col&&(gyid+i)<row)
            {
                reg[i][j]=input[(gyid+i)*col+gxid+j];
            }
        }
    }
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
        {
            if((gxid+j)<col&&(gyid+i)<row)
            {
                output[(gyid+i)*col+gxid+j]=reg[i][j];
            }
        }
    }
}
__global__ void memcpy_1d_no_hole(const float *input,float *output,size_t input_size)
{
    int tid=threadIdx.x+blockDim.x*blockIdx.x;
    if(tid<input_size)
    {
        output[tid]=input[tid];
    }
}
int main(int argc, char *argv[])
{
    if (argc != 3) {
    cerr<<"need 2 args"<<endl;
    exit(-1);
    }
    size_t M = strtoul(argv[1], NULL, 0);
    size_t N = strtoul(argv[2], NULL, 0);
    float *input=nullptr,*output=nullptr;
    CUDA_CHECK_MSG(cudaMalloc(&input,sizeof(float)*M*N),"cuda malloc:");
    CUDA_CHECK_MSG(cudaMalloc(&output,sizeof(float)*M*N),"cuda malloc:");
    dim3 block_init(256);
    dim3 grid_init(((M*N+4-1)/4+block_init.x-1)/block_init.x);
    generate_random<<<grid_init,block_init>>>(input,M*N,42);
    memcpy_1d_holed<<<grid_init,block_init>>>(input,output,M*N);
    memcpy_1d_lyl<<<grid_init,block_init>>>(input,output,M*N);
    memcpy_1d_holed_another<<<grid_init,block_init>>>(input,output,M*N);
    memcpy_1d_vector<<<grid_init,block_init>>>(input,output,M*N);
    dim3 grid_no_hole((M*N+block_init.x-1)/block_init.x);
    memcpy_1d_no_hole<<<grid_no_hole,block_init>>>(input,output,M*N);
    dim3 block_diffsuse(256);
    dim3 grid_diffsuse((((M*N+block_init.x-1)/block_init.x)+4-1)/4);
    memcpy_1d_diffsuse<<<grid_diffsuse,block_diffsuse>>>(input,output,M*N);
    dim3 block_2d(SM/4,SM/4);
    dim3 grid_2d(((N+4-1)/4+block_2d.x)/block_2d.x,((M+4-1)/4+block_2d.y-1)/block_2d.y);
    memcpy_2d_holed<<<grid_2d,block_2d>>>(input,output,M,N);
    memcpy_2d_holed_smem<<<grid_2d,block_2d>>>(input,output,M,N);
    memcpy_2d_holed_reg<<<grid_2d,block_2d>>>(input,output,M,N);
    cudaDeviceReset();
    return 0;
}