#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include "Type.hpp"

const static size_t TILE_WIDTH = 32;

// MatrixA size = kNum*matAyNum
// MatrixB size = matBxNum*kNum
// Output size = matBxNum*matAyNum
// call example matrixMultiply2D<<<{(matBxNum + threadIdx.x - 1)/threadIdx.x,(matAyNum + threadIdx.y -1)/threadIdx.y},{,}>>>
template<typename T>
__global__ static void matrixMultiply2DSharedMem(T* inputA, T* inputB, T* output, const size_t matAyNum, const size_t kNum, const size_t matBxNum)
{
    __shared__ T MA[TILE_WIDTH][TILE_WIDTH];
    __shared__ T MB[TILE_WIDTH][TILE_WIDTH];

    size_t matOutX =  blockIdx.x * blockDim.x + threadIdx.x;
    size_t matOutY =  blockIdx.y * blockDim.y + threadIdx.y;

    T retValue = 0;
    for (size_t tileId = 0; tileId < (kNum + TILE_WIDTH - 1) /TILE_WIDTH; ++tileId)
    {
        MA[threadIdx.y][threadIdx.x] = 0;
        MB[threadIdx.y][threadIdx.x] = 0;

        if (tileId * TILE_WIDTH + threadIdx.x < kNum && matOutY < matAyNum)
        {
            MA[threadIdx.y][threadIdx.x] = inputA[matOutY*kNum + tileId*TILE_WIDTH + threadIdx.x];
        }

        if( tileId * TILE_WIDTH + threadIdx.y < kNum && matOutX < matBxNum)
        {
            MB[threadIdx.y][threadIdx.x] = inputB[(tileId*TILE_WIDTH+threadIdx.y)*matBxNum + matOutX];
        }

        __syncthreads();

        for (size_t kid = 0; kid < TILE_WIDTH;++kid)
        {
            retValue += MA[threadIdx.y][kid] * MB[kid][threadIdx.x];
        }
        __syncthreads();
    }

    if (matOutX < matBxNum && matOutY < matAyNum)
    {
        output[matOutY * matBxNum + matOutX] = retValue;
    }
}

// MatrixA size = kNum*matAyNum
// MatrixB size = matBxNum*kNum
// Output size = matBxNum*matAyNum
// call example matrixMultiply2DBase<<<{(matBxNum + threadIdx.x - 1)/threadIdx.x,(matAyNum + threadIdx.y -1)/threadIdx.y},{,}>>>
template<typename T>
__global__ static void matrixMultiply2DBase(T* inputA, T* inputB, T* output, const size_t matAyNum, const size_t kNum, const size_t matBxNum)
{
    size_t elemX = blockIdx.x * blockDim.x + threadIdx.x;
    size_t elemY = blockIdx.y * blockDim.y + threadIdx.y;

    if (elemX >= matBxNum || elemY >= matAyNum)
    {
        return;
    }

    T retValue = 0;
    for (size_t i = 0;i < kNum; ++i)
    {
        retValue += inputA[elemY* kNum + i]*inputB[i* matBxNum + elemX];
    }
    output[elemY * matBxNum + elemX] = retValue;
}

template<typename T>
float matrixMultiply2DGPU( T* d_inputA, T* d_inputB, T* d_output, const size_t ARow, const size_t ACol, const size_t BRow, const size_t BCol, const MatrixAlgorithm algor)
{
    assert(ACol == BRow);
    const size_t kNum = ACol;
    dim3 grid((BCol + TILE_WIDTH - 1) / TILE_WIDTH,(ARow + TILE_WIDTH - 1) / TILE_WIDTH,1);
    dim3 block(TILE_WIDTH, TILE_WIDTH,1);
    switch (algor)
    {
        case MatrixAlgorithm::BASE:
        matrixMultiply2DBase<<<grid, block>>>(d_inputA, d_inputB, d_output,ARow, kNum, BCol);
        break;
        case MatrixAlgorithm::SHARED_MEM_BASE:
        matrixMultiply2DSharedMem<<<grid, block>>>(d_inputA, d_inputB, d_output,ARow,kNum, BCol);
        break;
        default:
        printf("[Matrix Muliply] Not have such algorithm.\n");
    }
    //CUDA_CHECK_ERROR_VOID();
    return 0;
}

template float matrixMultiply2DGPU<float>(float* d_inputA, float* d_inputB, float* d_output, const size_t ARow,const size_t ACol, const size_t BRow, const size_t BCol, const MatrixAlgorithm algor);
