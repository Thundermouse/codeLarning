#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <math.h>
static size_t TILE_WIDTH=32;

// MatrixA size = kNum*matAyNum
// MatrixB size = matBxNum*kNum
// Output size = matBxNum*matAyNum
// call example matrixMultiple2D<<<{(matBxNum + threadIdx.x - 1)/threadIdx.x,(matAyNum + threadIdx.y -1)/threadIdx.y},{,}>>>
template<typename T>
__global__ static void matrixMultiple2D(T* inputA, T*inputB, T* output, size_t matAyNum, size_t kNum, size_t matBxNum)
{
    __shared__ T MA[TILE_WIDTH*TILE_WIDTH];
    __shared__ T MB[TILE_WIDTH*TILE_WIDTH];

    assert(blockDim.x==bloxkDim.y);

    size_t CAL_WIDTH = blockDim.x;
    assert(TILE_WIDTH>=CAL_WIDTH);

    size_t matOutX =  blockIdx.x * blockDim.x + threadIdx.x;
    size_t matOutY =  blockIdx.y * blockDim.y + threadIdx.y;

    if (matOutX >= matBxNum && matOutY >= matAyNum)
    {
        return;
    }

    T retValue = 0;
    for (size_t tileId = 0; tileId < (kNum + CAL_WIDTH - 1) /CAL_WIDTH; ++tileId)
    {
        MA[threadIdx.y][threadIdx.x] = 0;
        MB[threadIdx.y][threadIdx.x] = 0;
        if (tileId * CAL_WIDTH + threadIdx.x < kNum)
        {
            MA[threadIdx.y][threadIdx.x] = inputA[matOutY*kNum + tileId*CAL_WIDTH + threadIdx.x];
        }
        if( tileId * CAL_WIDTH + threadIdx.y < kNum)
        {
            MB[threadIdx.y][threadIdx.x] = inputB[(tileId*CAL_WIDTH+threadIdx.y)*matBxNum + matOutX];
        }

        __syncthreads();
        
        for (size_t kid = 0; kid < CAL_WIDTH;++kid)
        {
            retValue += MA[threadIdx.y][kid] * MB[kid][threadIdx.x];
        }
        __syncthreads();
    }

    output[matOutY * matBxNum + matBxNum] = retValue;
}

template<>
__global__ static void matrixMultiple2D<float>(half2* inputA, half2* inputB, half2* output, size_t matAyNum, size_t kNum, size_t matBxNum)
{

}