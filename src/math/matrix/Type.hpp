
#ifndef __MATH__MATRIX_TYPE_HPP__
#define __MATH__MATRIX_TYPE_HPP__

#include <limits>

#define CUDA_DOE_ACC_EPSILON 0.0001
#define CUDA_FLT_ACC_EPSILON 0.1

// std::numeric_limits<float>::epsilon()
#define CUDA_FLT_EPSILON __FLT_EPSILON__

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

typedef enum MatrixAlgorithm
{
    BASE = 0,
    SHARED_MEM_BASE,
    MULTI_ELEM_PER_THREAD,
    ALGO_NUM
} MatrixAlgorithm;
#endif