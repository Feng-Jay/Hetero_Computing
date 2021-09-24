#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define PerThread 1024
#define N 64*256*PerThread//积分计算PI总共划分为这么多项相加
#define BlockNum 64 //block的数量
#define ThreadNum 256 //每个block中threads的数量

__global__ void Gpu_calPI(double* Gpu_list)
{
    int tid=blockIdx.x*blockDim.x*blockDim.y+threadIdx.x;
    int begin=tid*PerThread+1;
    int end=begin+PerThread;
    double temp=0;
    int flag=1;
    for(int i=begin;i<end;i++){
        temp+=flag*(1.0/(2*i-1));
        flag=flag*(-1);
    }
    Gpu_list[tid]=temp;
}

int main(void)
{
    double * cpu_list;
    double * Gpu_list;
    double outcome=0;
    cpu_list=(double*)malloc(sizeof(double)*BlockNum*ThreadNum);
    cudaMalloc((void**)&Gpu_list,sizeof(double)*BlockNum*ThreadNum);
    // dim3 blocksize=dim3(1,ThreadNum);
    // dim3 gridsize=dim3(1,BlockNum);

    Gpu_calPI<<<BlockNum,ThreadNum>>>(Gpu_list);

    cudaMemcpy(cpu_list,Gpu_list,sizeof(double)*BlockNum*ThreadNum,cudaMemcpyDeviceToHost);
    for(int i=0;i<BlockNum*ThreadNum;i++){
        outcome+=cpu_list[i];
    }
    outcome=4*outcome;
    printf("outcome=%.10f\n",outcome);
    // printf("block x=%d,y=%d\n",blocksize.x,blocksize.y);
    // printf("grid x=%d,y=%d\n",gridsize.x,gridsize.y);
    
}