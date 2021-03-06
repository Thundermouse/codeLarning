#include"BasicOperationCpuRef.cpp"
#include"BasicOperation.h"
#include<cuda_runtime_api.h>
#include<random>
#include<math.h>
#include<stdlib.h>
#include<stdint.h>
#include<assert.h>
#include<iostream>

//#define CUDA_CHECK_ERROR(func)                                                            \
//    {                                                                                     \
//        cudaError_t err = cudaGetLastError();                                             \
//        if (cudaSuccess != err)                                                           \
//        {                                                                                 \
//            std::runtime_error(std::string("[CUDA ERROR] In") + cudaGetErrorString(err) + \
//                               std::string("In Func: " #func) +                           \
//                               std::string("\n File: " __FILE__ ":") +                    \
//                               std::to_string(__LINE__));                                 \
//        }                                                                                 \
//    };

template<typename T>
bool compareArray(T* inputA, T*inputB, size_t eleNum)
{
    assert(std::is_trivial<T>::value && std::is_standard_layout<T>::value && "This function only be called with psd value.");
    for(size_t i = 0; i < eleNum; ++i)
    {
        if (inputA[i] != inputB[i])
        {
            printf("Array Element unequal at idx:%ld\n",i);
            return false;
        }
    }
    printf("Array Compare Pass!\n");
    return true;
}

int main()
{
    const size_t Row  = 1920;
    const size_t Col  = 1920;
    const size_t kNum = 1280;
    float* inputA    = new float[Row*kNum];    
    float* inputB    = new float[kNum*Col];  
    float* outputCPU = new float[Row*Col]; 
    float* outputGPU = new float[Row*Col]; 
    
    void* d_inputA = nullptr;
    void* d_inputB = nullptr;
    void* d_output = nullptr;

    cudaMalloc(&d_inputA, sizeof(float)*Row*kNum);
    cudaMalloc(&d_inputB, sizeof(float)*Col*kNum);
    cudaMalloc(&d_output, sizeof(float)*Row*Col);

    std::random_device seed;
    std::mt19937 gen(seed());
    std::uniform_real_distribution<> dis(0, 8.0);
    
    for (int j = 0; j< Row; ++j)
    {
        for (int i = 0; i < kNum; ++i)
        {
            inputA[j*kNum+i] = dis(gen);
        }
    }

    for (int j = 0; j< kNum; ++j)
    {
        for (int i = 0; i < Col; ++i)
        {
            inputB[j*Col+i] = dis(gen);
        }
    }

    cudaMemcpy(d_inputA, inputA, sizeof(float)*Row*kNum, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputB, inputB, sizeof(float)*Col*kNum, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, d_output, sizeof(float)*Row*Col, cudaMemcpyHostToDevice);

    matrixMultiple2DCPU(inputA, inputB, outputCPU, Row, kNum, Col);
    matrixMultiple2DGPU<float>(inputA, inputB, outputCPU, Row, kNum, kNum, Col);

    cudaMemcpy(outputGPU, d_output, sizeof(float)*Row*Col, cudaMemcpyDeviceToHost);

    compareArray(outputCPU, outputGPU, Row*Col);

    delete[] inputA;
    delete[] inputB;
    delete[] outputCPU;
    delete[] outputGPU;

    return 0;
}