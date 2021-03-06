#include <string>
#include <cuda_runtime.h>

#define CUDA_SAFTY_ERROR_CHECK false
// macro to easily check for cuda errors

#define cudaCheckError(func)                                                              \
    {                                                                                     \
        cudaError_t err = cudaGetLastError();                                             \
        if (cudaSuccess != err)                                                           \
        {                                                                                 \
            std::runtime_error(std::string("[CUDA ERROR] In") + cudaGetErrorString(err) + \
                               std::string("In Func: " #func) +                           \
                               std::string("\n File: " __FILE__ ":") +                    \
                               std::to_string(__LINE__));                                 \
        }                                                                                 \
    };

#define cudaSafetyCheckError(func)                                                                   \
    {                                                                                                \
        cudaCheckError(#func);                                                                       \
        err = cudaDeviceSynchronize();                                                               \
        if (cudaSuccess != err)                                                                      \
        {                                                                                            \
            std::runtime_error(std::string("[CUDA ERROR] Sync Error In") + cudaGetErrorString(err) + \
                               std::string("In Func: " #func) +                                      \
                               std::string("\n File: " __FILE__ ":") +                               \
                               std::to_string(__LINE__));                                            \
        }                                                                                            \
    };