#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define Vector_Power 20
#define Matrix_A_rows 10
#define Matrix_A_cols 10
#define Matrix_B_cols 10

#define TILE_WIDTH 16

// vector-ops.cpp
void add_vectors(float* a, float* b, float* c, int length, bool time);
float dot_product(float* a, float* b, int length, bool time);
void print_vector(float* a, int length, char name);
void sub_vectors(float* a, float* b, float* c, int length, bool time);

// matrix-ops.cpp
void add_matrices(float* a, float* b, float* c, int rows, int cols, bool time);
void mult_matrices(float* a, float* b, float* c, int rowsa, int colsa, int colsb, bool time);
void print_matrix(float* a, int rows, int cols, char name);
void transpose_matrix(float* a, float* c, int rows, int cols, bool time);
void sub_matrices(float* a, float* b, float* c, int rows, int cols, bool time);

// vector-ops.cu
__host__ void add_vectors_gpu(float* a, float* b, float* c, int length, bool time);
__host__ void dot_product_gpu_1block1(float* a, float* b, float* c, int length, bool time);
__host__ void dot_product_gpu_1block2(float* a, float* b, float* c, int length, bool time);
__host__ void dot_product_gpu_multiblock1(float* a, float* b, float* c, int length, bool time);
__host__ void dot_product_gpu_multiblock2(float* a, float* b, float* c, int length, bool time);
__host__ void sub_vectors_gpu(float* a, float* b, float* c, int length, bool time);

// matrix-ops.cu
__host__ void add_matrices_gpu(float* a, float* b, float* c, int rows, int cols, bool time);
__host__ void mult_matrices_gpu_niave1(float* a, float* b, float* c, int rowsa, int colsa, int colsb, bool time);
__host__ void mult_matrices_gpu_niave2(float* a, float* b, float* c, int rowsa, int colsa, int colsb, bool time);
__host__ void mult_matrices_gpu_tiled(float* a, float* b, float* c, int rowsa, int colsa, int colsb, bool time);
__host__ void transpose_matrices_gpu(float* src, float* dst, int rows, int cols, bool time);
__host__ void sub_matrices_gpu(float* a, float* b, float* c, int rows, int cols, bool time);