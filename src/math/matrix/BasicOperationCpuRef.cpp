#include<stdlib.h>
template<typename T>
void matrixMultiple2DCPU(T *inputA, T *inputB, T *output, size_t matAyNum, size_t kNum, size_t matBxNum)
{
    for (int j = 0; j < matAyNum; ++j)
    {
        for (int i = 0; i < matBxNum; ++i)
        {
            output[j*matBxNum+ i] = 0;
            for (int k = 0; k < kNum; ++k)
            {
                output[j*matBxNum+i] += inputA[j*kNum+k]*inputB[k*matBxNum+i];
            }
        }
    }
}