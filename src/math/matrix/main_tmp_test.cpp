#include "BasicOperationCpuRef.cpp"
#include "BasicOperation.h"
#include <cuda_runtime_api.h>
#include <random>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <iostream>

template <typename T>
bool compareArray(T *inputA, T *inputB, size_t eleNum)
{
    assert(std::is_trivial<T>::value && std::is_standard_layout<T>::value && "This function only be called with POD value.");
    std::cout.precision(10);
    for (size_t i = 0; i < eleNum; ++i)
    {
        if (std::fabs(inputA[i] - inputB[i]) > 0.00001)
        {
            std::cout << "Array Element unequal at idx:" << static_cast<int>(i) << ", A = " << inputA[i] << ", B = " << inputB[i] << ", delta = "<< inputA[i] - inputB[i]  <<std::endl;
            return false;
        }
    }
    printf("Array Compare Pass!\n");
    return true;
}

int main()
{
    const size_t ARow = 1920;
    const size_t BCol = 1920;
    const size_t kNum = 1280;
    float *inputA = new float[ARow * kNum];
    float *inputB = new float[kNum * BCol];
    float *outputCPU = new float[ARow * BCol];
    float *outputGPU = new float[ARow * BCol];

    float *d_inputA = nullptr;
    float *d_inputB = nullptr;
    float *d_output = nullptr;

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK_ERROR(cudaMalloc((void **)&d_inputA, sizeof(float) * ARow * kNum));
    CUDA_CHECK_ERROR(cudaMalloc((void **)&d_inputB, sizeof(float) * BCol * kNum));
    CUDA_CHECK_ERROR(cudaMalloc((void **)&d_output, sizeof(float) * ARow * BCol));

    std::random_device seed;
    std::mt19937 gen(seed());
    std::uniform_real_distribution<> dis(0, 8.0);

    for (int j = 0; j < ARow; ++j)
    {
        for (int i = 0; i < kNum; ++i)
        {
            inputA[j * kNum + i] = dis(gen);
        }
    }

    for (int j = 0; j < kNum; ++j)
    {
        for (int i = 0; i < BCol; ++i)
        {
            inputB[j * BCol + i] = dis(gen);
        }
    }
    CUDA_CHECK_ERROR(cudaEventCreate(&start));
    CUDA_CHECK_ERROR(cudaEventCreate(&stop));
    CUDA_CHECK_ERROR(cudaMemcpy(d_inputA, inputA, sizeof(float) * ARow * kNum, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_inputB, inputB, sizeof(float) * BCol * kNum, cudaMemcpyHostToDevice));

    matrixMultiply2DCPU(inputA, inputB, outputCPU, ARow, kNum, BCol);

    CUDA_CHECK_ERROR(cudaEventRecord(start));
    matrixMultiply2DGPU(d_inputA, d_inputB, d_output, ARow, kNum, kNum, BCol, MatrixAlgorithm::SHARED_MEM_BASE);
    CUDA_CHECK_ERROR(cudaEventRecord(stop));

    CUDA_CHECK_ERROR(cudaMemcpy(outputGPU, d_output, sizeof(float) * ARow * BCol, cudaMemcpyDeviceToHost));

    float ElpaseTimeMs = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ElpaseTimeMs, start, stop);
    std::cout<<"Kernel Elapsed Time : "<< ElpaseTimeMs << "ms"<<std::endl;
    compareArray(outputCPU, outputGPU, ARow * BCol);

    CUDA_CHECK_ERROR(cudaFree(d_inputA));
    CUDA_CHECK_ERROR(cudaFree(d_inputB));
    CUDA_CHECK_ERROR(cudaFree(d_output));

    delete[] inputA;
    delete[] inputB;
    delete[] outputCPU;
    delete[] outputGPU;

    return 0;
}
