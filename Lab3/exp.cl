__kernel void VectorAdd(__global const double* a, 
                        __global const double* b, 
                        __global double* c)
{
    int iGID = get_global_id(0);
    c[iGID]= a[iGID] + b[iGID]; 
}