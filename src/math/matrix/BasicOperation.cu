#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <assert.h>
const static size_t TILE_WIDTH = 32;

// MatrixA size = kNum*matAyNum
// MatrixB size = matBxNum*kNum
// Output size = matBxNum*matAyNum
// call example matrixMultiple2D<<<{(matBxNum + threadIdx.x - 1)/threadIdx.x,(matAyNum + threadIdx.y -1)/threadIdx.y},{,}>>>
template<typename T>
__global__ static void matrixMultiple2D(T* inputA, T* inputB, T* output, const size_t matAyNum, const size_t kNum, const size_t matBxNum)
{
    __shared__ T MA[TILE_WIDTH][TILE_WIDTH];
    __shared__ T MB[TILE_WIDTH][TILE_WIDTH];


    const size_t CAL_WIDTH = blockDim.x;
    
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


template<typename T>
float matrixMultiple2DGPU(T* d_inputA, T* d_inputB, T* d_output, const size_t ARow, const size_t ACol, const size_t BRow, const size_t BCol)
{
    assert(ACol == BRow);
    const size_t kNum = ACol;
    dim3 grid((BCol + TILE_WIDTH - 1) / TILE_WIDTH,(BCol + TILE_WIDTH - 1) / TILE_WIDTH,1);
    dim3 block(TILE_WIDTH, TILE_WIDTH,1);
    matrixMultiple2D<<<grid, block>>>(d_inputA, d_inputB, d_output,kNum, ARow, BCol);
    return 0;
}

template float matrixMultiple2DGPU<float>(float* d_inputA, float* d_inputB, float* d_output, const size_t ARow,const size_t ACol, const size_t BRow, const size_t BCol);
