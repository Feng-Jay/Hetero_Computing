#include <stdio.h>
#include <time.h>

#define M 32//矩阵维度
#define Exp 4096//计算的幂次

__constant__ const int gpu_m=M;
__constant__ const int gpu_exp=Exp;

__global__ void Gpu_FastExp(float *gpu_martix,float* gpu_res)
{
    // printf("%d ", threadIdx.y)
    __shared__ float temp[gpu_m*gpu_m];
    float res[gpu_m];
    int tid=threadIdx.y;
    temp[tid]=gpu_martix[tid];
    __syncthreads();
    for(int i=1;i<gpu_exp;i*=2){
        for(int k=0;k<M;k++)
        {
            res[k]=0;
            for(int j=0;j<M;j++){
                res[k]+=temp[tid*M+j]*temp[j*M+tid];
            }
        }
        for(int i=0;i<M;i++){
        gpu_res[tid*M+i]=res[i];
        temp[tid*M+i]=res[i];
        }
        __syncthreads();
    }
}

int main()
{
    float * cpu_martix;
    float * cpu_res;
    float* gpu_martix;
    float * gpu_res;
    cudaMallocHost((void**)&cpu_martix,M*M*sizeof(float));
    cudaMallocHost((void**)&cpu_res,M*M*sizeof(float));
    cudaMalloc((void**)&gpu_res,M*M*sizeof(float));
    cudaMalloc((void**)&gpu_martix,M*M*sizeof(float));
    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            cpu_martix[i*M+j] = 1.0/M;
        }
    }
    printf("init elem=%f",1.0/M);
    cudaMemcpy(gpu_martix,cpu_martix,M*M*sizeof(float),cudaMemcpyHostToDevice);
    float begin=clock();
    dim3 threads(1,M);
    // printf("一切正常\n");
    Gpu_FastExp<<<1,threads>>>(gpu_martix,gpu_res);
    cudaDeviceSynchronize();
    // printf("一切正常\n");
    cudaMemcpy(cpu_res,gpu_res,M*M*sizeof(float),cudaMemcpyDeviceToHost);
    float end=clock();
    printf("result = %.10f time spend=%.10f\n",cpu_res[0],(end-begin)/CLOCKS_PER_SEC);
}
