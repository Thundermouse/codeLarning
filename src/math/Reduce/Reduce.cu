#include <cuda_runtime_api.h>
#include <cooperative_groups.h>

static constexpr MAX_THREAD_PER_BLOCK = 1024;
static constexpr DEFAULT_WARP_SIZE = 32;

template<typename T, size_t threadNum>
__device__ __forceinline__ void lastWarpReduce(volatile T* sharedMem, uint32_t tid)
{
 
        if( threadNum >= 64 ) val[tIdx] += val[tIdx + 32];
        if( threadNum >= 32 ) val[tIdx] += val[tIdx + 16];
        if( threadNum >= 16 ) val[tIdx] += val[tIdx + 8];
        if( threadNum >= 8  ) val[tIdx] += val[tIdx + 4];
        if( threadNum >= 4  ) val[tIdx] += val[tIdx + 2];
        if( threadNum >= 2  ) val[tIdx] += val[tIdx + 1];
}

template<typename T>
__device__ __forceinline__ void lastWarpReduce<T,0>(volatile T* sharedMem, uint32_t tid)
{
        val[tIdx] += val[tIdx + 32];
        val[tIdx] += val[tIdx + 16];
        val[tIdx] += val[tIdx + 8];
        val[tIdx] += val[tIdx + 4];
        val[tIdx] += val[tIdx + 2];
        val[tIdx] += val[tIdx + 1];
}

// block reduce only use one cudaBlock to reduce element.
// Advantage    : one kernel to solve the problem
// Disadvantage : only have one block, which means only use one SM!
// [src]: inputParam, array address
// [size]: inputParam, array size
template<typename T>
__device__ T blockReduce(T* src, uint32_t size)
{
    assert( blockIdx.x < 1, "[blockReduce] block reduce can only have one block.")
    assert( !(blockDim.x & (blockDim.x-1)), "[blockReduc] blockDim must be power of 2.")

    const uint32_t tIdx = threadIdx.x;
    
    __shared__ T val[MAX_THREAD_PER_BLOCK];
    uint32_t threadElemNum = blockDim.x;

    // initialize with tail element;
    if (tIdx < size % threadElemNum)
    {
        val[tIdx] = src[size - tIdx)];
    }
    else 
    {
        val[tIdx] = 0;
    }
    size -= size % threadElemNum;

    for (;size>threadElemNum; size -= threadElemNum)
    {
        val[tIdx] += src[size - tIdx];
    }
    __syncthreads();

    for( threadElemNum >>= 1 ; threadElemNum > 32; threadElemNum >>= 1) )
    {
        if (tIdx < threadElemNum)
        {
            val[tIdx] += val[tIdx + threadElemNum];
        }
        __syncthreads();
    }

    if (tIdx < DEFAULT_WARP_SIZE)
    {
        lastWarpReduce(val, tIdx);
    }
    return val[0];
}

// reduce can use multi cudaBlock.
// call kernel: reduce<<<(size + (threadPerBlock.x * times) -1 )/(threadPerBlock*times), threadPerBlock>>>(arr,size,tines,dst)
template<typename T>
__device__ void reduce<T, 0>(const T* src, const uint32_t size, const uint32_t loadTimes, T* dst)
{
    __shared__ T val[MAX_THREAD_PER_BLOCK];

    const uint32_t tIdx = blockIdx.x * blockDim.x +  threadIdx.x;
    uint32_t loadStartIdx = blockIdx.x * blockDim.x * loadTimes * 2;

    // loading element by loading time
    val[tIdx] = 0;
    for (uint32_t loadRound = 0;  loadRound < loadTimes && loadStartIdx < size;  loadStartIdx += blockDim.x, ++loadRound)
    {
        const uint32_t arrayIdx = loadStartIdx + tidx;
        if (arrayIdx < size)
        {
            val[tIdx] += src[arrayIdx];
        }
    }
    __syncthreads();
    // reduce in shared mem
    for (uint32_t STEP = (blockDim.x >> 1); STEP > 32; STEP >>= 1)
    {
        if (tIdx < STEP)
        {
            val[tIdx] = val[tIdx + STEP];
        }
        __syncthreads();
    }

    if (tIdx < DEFAULT_WARP_SIZE)
    {
        lastWarpReduce<T,0>(val, tIdx);
    }

    if(threadIdx.x == 0) dst[blockIdx.x] = val[0];
    return;
}

// use template to unroll every thing;
template<typename T, size_t threadNum>
__device__ void reduce(const T* src, const uint32_t size, const uint32_t loadTimes, T* dst)
{
    __shared__ T val[MAX_THREAD_PER_BLOCK];

    const uint32_t tIdx = blockIdx.x * blockDim.x +  threadIdx.x;
    uint32_t loadStartIdx = blockIdx.x * blockDim.x * loadTimes;

    // loading element by loading time
    val[tIdx] = 0;
    for (uint32_t loadRound = 0;  loadRound < loadTimes && loadStartIdx < size;  loadStartIdx += blockDim.x, ++loadRound)
    {
        const uint32_t arrayIdx = loadStartIdx + tidx;
        if (arrayIdx < size)
        {
            val[tIdx] += src[arrayIdx];
        }
    }
    __syncthreads();

    // reduce in shared mem
    #pragma unroll
    for (uint32_t STEP = (blockDim.x >> 1); STEP > 32; STEP >>= 1)
    {
        if (tIdx < STEP)
        {
            val[tIdx] = val[tIdx + STEP];
        }
        __syncthreads();
    }

    // reduce within warp
    if (tIdx < DEFAULT_WARP_SIZE)
    {
        lastWarpReduce<T, threadNum>(val, tIdx);
    }

    if(threadIdx.x == 0) dst[blockIdx.x] = val[0];
    return;
}

// reduce by warp level;
template<typename T, size_t threadNum>
__device__ void reduce(const T* src, const uint32_t size, const uint32_t loadTimes, T* dst)
{
    assert( !(loadTimes & 0x1), "[reduce] loadTimes must be odd." )

    static constexpr MAX_WARP_PER_BLOCK = MAX_THREAD_PER_BLOCK / DEFAULT_WARP_SIZE;
    __shared__ T warpVal[MAX_WARP_PER_BLOCK];

    uint32_t warpId = threadIdx.x / DEFAULT_WARP_SIZE;
    bool isMasterLane = threadIdx.x % DEFAULT_WARP_SIZE == 0 ? true:false;

    if (isMasterLane)
    {
        warpVal[warpId] = 0;
    }

    // load and reduce
    uint32_t loadStartIdx = blockIdx.x * blockDim.x * loadTimes;
    for (uint32_t loadRound = 0;  loadRound < loadTimes && loadStartIdx < size;  loadStartIdx += blockDim.x, ++loadRound)
    {
        uint32_t loadIdx = loadStartIdx + tidx;
        unsigned mask = __ballot_sync(0xFFFFFFFF, loadIdx < size);
        if (loadIdx < size)
        {
            T val = src[loadIdx];
            #pragma unroll
            for ( uint32_t STEP = DEFAULT_WARP_SIZE / 2; STEP > 0; STEP >>= 2)
                val += __shfl_down_sync(mask, val, STEP);
        }
        
        if (isMasterLane) warpVal[warpId] = val;
    }

    __syncthreads();
    // reduce in shared memroy within a warp
    if (tIdx < DEFAULT_WARP_SIZE)
    {
        T val = warpVal[tIdx];
        #pragma unroll
        for ( uint32_t STEP = DEFAULT_WARP_SIZE / 2; STEP > 0; STEP >>= 2)
            val += __shfl_down_sync(mask, val, STEP);
    }

    if(isMasterLane) dst[blockIdx.x] = val;
    return;
}
