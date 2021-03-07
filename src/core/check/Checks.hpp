#ifndef __CORE_CHECK__
#define __CORE_CHECK__

#include <string>
#include <cuda_runtime_api.h>

#define CUDA_CHECK_ERROR(func)                                                                 \
    {                                                                                          \
        cudaError_t err = cudaGetLastError();                                                  \
        if (cudaSuccess != err)                                                                \
        {                                                                                      \
            std::cout << "[CUDA ERROR] Last Run In"                                            \
                      << " File: " << __FILE__ << ":" << __LINE__ << ":" << #func << std::endl \
                      << " ErrorStr : " << cudaGetErrorString(err) << std::endl;               \
            exit(-1);                                                                          \
        };                                                                                     \
        err = func;                                                                            \
        if (cudaSuccess != err)                                                                \
        {                                                                                      \
            std::cout << "[CUDA ERROR] Current Func"                                           \
                      << " File: " << __FILE__ << ":" << __LINE__ << ":" << #func << std::endl \
                      << " ErrorStr: " << cudaGetErrorString(err) << std::endl;                \
            exit(-1);                                                                          \
        };                                                                                     \
    }

// TODO use print
//#define CUDA_CHECK_ERROR_VOID()\
//    {                                                                                           \
//        cudaError_t err = cudaGetLastError();                                                   \
//        if (cudaSuccess != err)                                                                 \
//        {                                                                                       \
//            std::cout << "[CUDA ERROR] Last Run In"                                             \
//                      << " File: " << __FILE__ << ":" << __LINE__ << std::endl; \
//            exit(-1);                                                                           \
//        };                                                                                      \                                                                                   \
//    }

#endif