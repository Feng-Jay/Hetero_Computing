#include <stdio.h>
#include <time.h>

#define N 99//矩阵的N次方
// 99=64+32+2+1
#define BlockNum 1//block的数量
#define ThreadNum 4 //每个block中threads的数量
#define m 32//每个行有多少个元素，即矩阵的维度

__global__ void Gpu_func(double* gpu_martix, double* gpu_res,int* exp)
{
   int tid=threadIdx.x;
   int index=tid*m*m;
   printf("tid=%d exp=%d\n",tid,exp[tid]);
   __shared__ double cache[m*m*3];
   for(int i=0;i<exp[tid];i++){
      for(int p=0;p<m;p++){
         for(int q=0;q<m;q++){
            for(int k=0;k<m;k++){
               gpu_res[p*m+q]+=gpu_martix[p*m+k]*gpu_martix[k*m+q];
            }
         }
      }
      for(int i=0;i<m;i++){
         for(int j=0;j<m;j++){
            gpu_martix[i*m+j]=gpu_res[i*m+j];
            gpu_res[i*m+j]=0;
         }
      }
   }
   if(tid!=0){
    for(int i=0;i<m;i++){
         for(int j=0;j<m;j++){
            cache[(tid-1)*m*m+i*m+j]=gpu_martix[i*m+j];
            __syncthreads();
         }
      }
   }
   if(tid==0){
      for(int num=0;num<3;num++){
         for(int i=0;i<m;i++){
            for(int j=0;j<m;j++){
               for(int k=0;k<m;k++){
                  gpu_res[i*m+j]+=gpu_martix[i*m+k]*cache[num*m*m+k*m+j]; 
               }
            }
         }
         if(num<2){
            for(int i=0;i<m;i++){
               for(int j=0;j<m;j++){
                  gpu_martix[i*m+j]=gpu_res[i*m+j];
                  gpu_res[i*m+j]=0;
               }
            }
         }
      }
   }

}

int 
main()
{
   double* cpu_martix;
   double* gpu_martix;
   double* cpu_res;
   double* gpu_res;
   int* gpu_exp;
   cpu_martix=(double*)malloc(sizeof(double)*m*m);
   cpu_res=(double*)malloc(sizeof(double)*m*m);
   for(int i=0;i<m*m;i++){
       cpu_martix[i]=1.0/(m);
       cpu_res[i]=0;
   }
   int exp[4]={6,5,1,0};
   printf("init elem is %f",1.0/(m));
   printf("\n");
   double begin=clock();
   cudaMalloc((void**)&gpu_martix,sizeof(double)*m*m);
   cudaMalloc((void**)&gpu_res,sizeof(double)*m*m);
   cudaMalloc((void**)&gpu_exp,sizeof(int)*4);
   cudaMemcpy(gpu_martix,cpu_martix,sizeof(double)*m*m,cudaMemcpyHostToDevice);
   cudaMemcpy(gpu_res,cpu_res,sizeof(double)*m*m,cudaMemcpyHostToDevice);
   cudaMemcpy(gpu_exp,exp,sizeof(int)*4,cudaMemcpyHostToDevice);

   Gpu_func<<<BlockNum,ThreadNum>>>(gpu_martix,gpu_res,gpu_exp);
   cudaMemcpy(cpu_res,gpu_res,sizeof(double)*m*m,cudaMemcpyDeviceToHost);
   double end=clock();
    printf("spend %.10f s\n",(end-begin)/(CLOCKS_PER_SEC));
    printf("here is outcome martix......................\n");
    printf("%.20f\n",cpu_res[0]);
}