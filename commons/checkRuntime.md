

```c++
#include <stdio.h>
#include <cuda_runtime.h>

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)
bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line)
{
    if(code != cudaSuccess){
        const char* err_name = cudaGetErrorName(code);
        const char* err_str  = cudaGetErrorString(code);
        printf("CUDA Runtime Error [%s: %d] %s failed. \n Error Name: %s, Error String: %s.\n",
            file, line, op, err_name, err_str
        );
        return false;
    }
    return true;
}
```