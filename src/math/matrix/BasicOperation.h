#include<stdlib.h>
#include "Type.hpp"
template<typename T>
float matrixMultiply2DGPU(T* d_inputA,T* d_inputB, T* d_output, const size_t ARow, const size_t ACol,const size_t BRow,const size_t BCol, const MatrixAlgorithm algor);

//float matrixMultiply2DGPU(float* d_inputA,float* d_inputB, float* d_output, const size_t ARow, const size_t ACol,const size_t BRow,const size_t BCol);