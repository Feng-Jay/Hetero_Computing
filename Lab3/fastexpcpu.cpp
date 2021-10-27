#include <iostream>
#include <memory>
#include <stdio.h>
#include <time.h>
#include <cstring>
#include <CL/opencl.h>

using namespace std;
int num=10;
int exp=128;

int main(int argc, char* argv[])
{
    cl_int err;
    cl_platform_id platfrom;
    err=clGetPlatformIDs(1,&platfrom,nullptr);
    if(err!=CL_SUCCESS){
        std::cout<<"Can't select a platfrom"<<std::endl;
        return -1;
    }
    cl_device_id device;
    err=clGetDeviceIDs(platfrom,CL_DEVICE_TYPE_CPU,1,&device,nullptr);
    if(err!=CL_SUCCESS){
        std::cout<<"Can't select a device"<<std::endl;
        return -1;
    }
    cl_context context;
    context=clCreateContext(nullptr,1,&device,nullptr,nullptr,&err);
    if(err!=CL_SUCCESS){
        std::cout<<"Can't create proper context"<<std::endl;
        return -1;
    }
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if(err!=CL_SUCCESS){
        std::cout<<"Create command queue failed"<<std::endl;
        return -1;
    }
    double* hA = new double[num*num];
    double* hB = new double[num*num];
    double* hC = new double[num*num];
    memset(hC, 0, sizeof(double)*num);
    for(int i=0;i<num*num;i++){
        hA[i]=hB[i]=1.0/num;
    }
    cl_mem mA = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(double) * num*num, hA, nullptr);
    cl_mem mB= clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(double) * num*num, hB, nullptr);
    cl_mem mC = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * num*num,nullptr, nullptr);
    // cl_mem mnum = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(int), &num, nullptr);
    // cl_mem mexp = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(int), &exp, nullptr); 
    if (mA == nullptr || mB == nullptr || mC == nullptr) {
        std::cout << "Create buffer failed" << std::endl;
        return -1;
    }
    const char* program_source =
    "__kernel void test_main(__global double* A, __global const double* B, __global double* C, int num, int exp) {\n"
    "  size_t idx = get_global_id(0);\n"
    "  __local double localbuffer[num*num];\n"
    "  localbuffer[idx]=A[idx];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  double temp=0;\n"
    "  for(int i=1;i<exp;i*=2){\n"
            "temp=0;\n"
            "for(int j=0;j<num;j++) temp+=localbuffer[(idx/num)*num+j]*localbuffer[j*num+(idx%num)];\n" 
            "localbuffer[idx]=temp;\n"  
            "C[idx]=temp;\n" 
            "barrier(CLK_LOCAL_MEM_FENCE);\n"
        "}\n"
    "}";
    cl_program program = clCreateProgramWithSource(context, 1, &program_source, nullptr, nullptr);
    if(program==nullptr){
        std:: cout << "Create Program failed!" << std:: endl;
        return -1;
    }
    err=clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Build program failed" << std::endl;
        size_t length;
        char buffer[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
        cout<<"--- Build log ---"<<buffer<<endl;
        return -1;
    }
    cl_kernel kernel = clCreateKernel(program, "test_main", nullptr);
    if (kernel == nullptr) {
        std::cout << "Create kernel failed" << std::endl;
        return -1;
    }
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mA);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &mB);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &mC);
    err |= clSetKernelArg(kernel, 3, sizeof(int), (void*)&num);
    err |= clSetKernelArg(kernel, 4, sizeof(int), (void*)&exp);
    if (err != CL_SUCCESS) {
        std::cout << "Set kernel arg failed" << std::endl;
        return -1;
    }
    cout<<"Before exp\n";
    for(int i=0;i<10;i++){
        for(int j=0;j<10;j++) std:: cout<<hA[i*10+j]<<" ";
        std:: cout<< std:: endl;
    }
    size_t globalWorkSize[]{100};
    size_t localWorkSize[] {100};
    double begin=clock();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Run kernel failed" << std::endl;
        return -1;
    }
    double end=clock();
    err = clEnqueueReadBuffer(queue, mC, CL_TRUE, 0, sizeof(double) * num*num, hC, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Read data failed" << std::endl;
        return -1;
    }
    cout<<"After exp\n";
    for(int i=0;i<10;i++){
        for(int j=0;j<10;j++) std:: cout<<hC[i*10+j]<<" ";
        std:: cout<< std:: endl;
    }
    printf("Time spend= %.10f\n", (end-begin)/CLOCKS_PER_SEC);
    // check one output data
    // if (hC[1024] != hA[1024] + hB[1024]) {
    //     std::cout << "Data calculation failed" << std::endl;
    //     return -1;
    // }
    return 0;
}