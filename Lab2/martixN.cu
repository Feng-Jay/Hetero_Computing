#include <stdio.h>
#include <time.h>

#define N 64//矩阵的N次方
#define BlockNum 8//block的数量
#define ThreadNum 64 //每个block中threads的数量
#define m 32//每个行有多少个元素，即矩阵的维度

__global__ void Gpu_martixN(double* Gpu_martix, double* Gpu_res)
{
    //每个GPU核函数计算矩阵的一行
    int tid=blockIdx.x*blockDim.x*blockDim.y+threadIdx.x;
    double temp[m];
    for(int i=0;i<m;i++) temp[i]=0;

    for(int exp=0;exp<N;exp++){
        for(int i=0;i<m;i++){
            for(int j=0;j<m;j++){
                temp[i]+=Gpu_res[tid*m+j]*Gpu_martix[j*m+i];
            }//算完1个
        }//算完1行
        for(int i=0;i<m;i++) {Gpu_res[tid*m+i]=temp[i];temp[i]=0;}
    }
}

int main()
{
    double* Cpu_martix;
    double* Cpu_res;
    double* Gpu_martix;
    double* Gpu_res;
    Cpu_martix=(double*)malloc(sizeof(double)*m*m);
    Cpu_res=(double*)malloc(sizeof(double)*m*m);
    //初始化矩阵
    for(int i=0;i<m*m;i++){
       Cpu_martix[i]=1.0/(m);
       Cpu_res[i]=1.0/(m);
    }
    printf("init elem is %f",1.0/(m));
    printf("\n");
    cudaMalloc((void**)&Gpu_martix,sizeof(double)*m*m);
    cudaMalloc((void**)&Gpu_res,sizeof(double)*m*m);
    cudaMemcpy(Gpu_martix,Cpu_martix,sizeof(double)*m*m,cudaMemcpyHostToDevice);
    cudaMemcpy(Gpu_res,Cpu_res,sizeof(double)*m*m,cudaMemcpyHostToDevice);
    printf("begin exec\n");
    double begin=clock();
    Gpu_martixN<<<BlockNum,ThreadNum>>>(Gpu_martix,Gpu_res);
    double end=clock();
    cudaMemcpy(Cpu_res,Gpu_res,sizeof(double)*m*m,cudaMemcpyDeviceToHost);
    printf("spend %.10f s\n",(end-begin)/(CLOCKS_PER_SEC));
    printf("here is outcome martix......................\n");
    printf("%f\n",Cpu_res[0]);
}