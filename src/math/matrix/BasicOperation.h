#include<stdlib.h>

template<typename T>
float matrixMultiple2DGPU(T* d_inputA, T* d_inputB, T* d_output, const size_t ARow, const size_t ACol,const size_t BRow,const size_t BCol);
