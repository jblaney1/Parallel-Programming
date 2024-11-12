/*
* Author: Josh Blaney
* Date: 11/04/2024
* 
* Description:
* 
* 
*/


#include <iostream>
#include <random>
#include "Linear Algebra.h"


void initialize_array(float* a, int length, float max, float min) {
    /*
    * Populates an array (a) of known length (length) with random
    * floats in a provided range (min to max), in place.
    */

    float range = max - min;

    for (int i = 0; i < length; i++) {
        a[i] = (float(rand()) / float(RAND_MAX)) * range + min;
    }
}


void validate_results(float* a, float* b, int length, float tolerance) {
    /*
    * Checks two arrays (a and b) for inconsistencies which are beyond
    * a specified tolerance (tolerance) value. If an inconsistency is
    * found, the function prints a warning and exits.
    */

    float difference;

    for (int i = 0; i < length; i++) {
        difference = abs(a[i] - b[i]);
        if (difference > tolerance) { 
            std::cout << "[WARNING] The results are not within tolerance: (A) " << a[i] << " and (B) " << b[i] << " | " << difference << "\n\n";
            break; 
        }
    }
}


int main()
{
    srand(time (NULL));

    // Parameters
    int length = 1 << Vector_Power;   // Vector length
    int rowsa = 1 << Matrix_A_rows;    // Matrix A rows
    int colsa = 1 << Matrix_A_cols;    // Matrix A columns and Matrix B rows
    int colsb = 1 << Matrix_B_cols;    // Matrix B columns

    float max = 1.0f;       // The maximum floating point value for initialization
    float min = -1.0;       // The minimum floating point value for initialization

    int timing_size = 2 << 8;   // Only print the timing info for runs with more parameters than this
    int cpu_limit = 2 << 24;    // Only run the CPU version if it will complete in a reasonable time
    float tolerance = 5E-4;     // Validation tolerance

    // Vectors
    float* vec_a = new float[length]();     // Source vector a
    float* vec_b = new float[length]();     // Source vector b
    float* hvec_c = new float[length]();    // Destination vector c in host memory
    float* dvec_c = new float[length]();    // Destination vector c in device memory

    float h_c = 0.0f;                       // Dot product result in host memory

    // Matrices
    float* mat_a = new float[rowsa * colsa]();  // Source matrix a
    float* mat_b = new float[colsa * colsb]();  // Source matrix b
    float* hmat_c = new float[rowsa * colsb](); // Destination matrix c in host memory
    float* dmat_c = new float[rowsa * colsb](); // Destination matrix c in device memory

    /*
    * -------------- Vector Operations --------------
    */
    initialize_array(vec_a, length, max, min);
    initialize_array(vec_b, length, max, min);

    std::cout << "Performing operations on vectors of length: " << length << "\n";
    print_vector(vec_a, length, 'A');
    print_vector(vec_b, length, 'B');

    // Vector Addition
    add_vectors(vec_a, vec_b, hvec_c, length, length >= timing_size);
    print_vector(hvec_c, length, 'C');

    add_vectors_gpu(vec_a, vec_b, dvec_c, length, length >= timing_size);
    print_vector(hvec_c, length, 'C');

    validate_results(hvec_c, dvec_c, length, tolerance);

    // Vector Subtraction
    sub_vectors(vec_a, vec_b, hvec_c, length, length >= timing_size);
    print_vector(hvec_c, length, 'C');

    sub_vectors_gpu(vec_a, vec_b, dvec_c, length, length >= timing_size);
    print_vector(hvec_c, length, 'C');

    validate_results(hvec_c, dvec_c, length, tolerance);

    // Dot Product
    h_c = dot_product(vec_a, vec_b, length, length >= timing_size);
    std::cout << "Printing Dot Product Result (CPU)\n\t" << h_c << "\n\n";

    initialize_array(dvec_c, length, 0.0f, 0.0f);
    dot_product_gpu_1block1(vec_a, vec_b, dvec_c, length, length >= timing_size);
    std::cout << "Printing Dot Product Result (1Block Niave GPU)\n\t" << dvec_c[0] << "\n\n";

    initialize_array(dvec_c, length, 0.0f, 0.0f);
    dot_product_gpu_1block2(vec_a, vec_b, dvec_c, length, length >= timing_size);
    std::cout << "Printing Dot Product Result (1Block Coallesced GPU)\n\t" << dvec_c[0] << "\n\n";

    initialize_array(dvec_c, length, 0.0f, 0.0f);
    dot_product_gpu_multiblock1(vec_a, vec_b, dvec_c, length, length >= timing_size);
    std::cout << "Printing Dot Product Result (MultiBlock Niave GPU)\n\t" << dvec_c[0] << "\n\n";

    initialize_array(dvec_c, length, 0.0f, 0.0f);
    dot_product_gpu_multiblock2(vec_a, vec_b, dvec_c, length, length >= timing_size);
    std::cout << "Printing Dot Product Result (MultiBlock Coallesced GPU)\n\t" << dvec_c[0] << "\n\n";

    /*
    * -------------- Matrix Operations --------------
    */
    initialize_array(mat_a, rowsa * colsa, max, min);
    initialize_array(mat_b, colsa * colsb, max, min);

    std::cout << "Performing matrix operations on matrices of sizes: (A) ~ " << rowsa << "x" << colsa << " and (B) ~ " << colsa << "x" << colsb << "\n";
    print_matrix(mat_a, rowsa, colsa, 'A');
    print_matrix(mat_b, colsa, colsb, 'B');

    // Matrix Addition
    if (rowsa == colsa && colsa == colsb) {
        add_matrices(mat_a, mat_b, hmat_c, rowsa, colsa, rowsa * colsa >= timing_size);
        print_matrix(hmat_c, rowsa, colsb, 'C');

        add_matrices_gpu(mat_a, mat_b, dmat_c, rowsa, colsa, rowsa * colsa >= timing_size);
        print_matrix(dmat_c, rowsa, colsb, 'C');

        validate_results(hmat_c, dmat_c, rowsa * colsb, tolerance);
    }

    // Matrix Subtraction
    if (rowsa == colsa && colsa == colsb) {
        sub_matrices(mat_a, mat_b, hmat_c, rowsa, colsa, rowsa * colsa >= timing_size);
        print_matrix(hmat_c, rowsa, colsb, 'C');

        sub_matrices_gpu(mat_a, mat_b, dmat_c, rowsa, colsa, rowsa * colsa >= timing_size);
        print_matrix(dmat_c, rowsa, colsb, 'C');

        validate_results(hmat_c, dmat_c, rowsa * colsb, tolerance);
    }

    // Matrix Multiplication
    if (rowsa * colsa < cpu_limit && colsa * colsb < cpu_limit) {
        initialize_array(hmat_c, rowsa * colsb, 0.0f, 0.0f);
        mult_matrices(mat_a, mat_b, hmat_c, rowsa, colsa, colsb, rowsa * colsa >= timing_size);
        print_matrix(hmat_c, rowsa, colsb, 'C');
    }
    else {
        std::cout << "[WARNING] The matrices are too large to compute in a reasonable time on the CPU, try the GPU instead!\n\n";
    }

    initialize_array(dmat_c, rowsa * colsb, 0.0f, 0.0f);
    mult_matrices_gpu_niave1(mat_a, mat_b, dmat_c, rowsa, colsa, colsb, rowsa * colsa >= timing_size);
    print_matrix(dmat_c, rowsa, colsb, 'C');

    if (rowsa * colsa < cpu_limit && colsa * colsb < cpu_limit) {
        validate_results(hmat_c, dmat_c, rowsa * colsb, tolerance);
    }

    initialize_array(dmat_c, rowsa * colsb, 0.0f, 0.0f);
    mult_matrices_gpu_niave2(mat_a, mat_b, dmat_c, rowsa, colsa, colsb, rowsa * colsa >= timing_size);
    print_matrix(dmat_c, rowsa, colsb, 'C');

    if (rowsa * colsa < cpu_limit && colsa * colsb < cpu_limit) {
        validate_results(hmat_c, dmat_c, rowsa * colsb, tolerance);
    }

    initialize_array(dmat_c, rowsa * colsb, 0.0f, 0.0f);
    mult_matrices_gpu_tiled(mat_a, mat_b, dmat_c, rowsa, colsa, colsb, rowsa * colsa >= timing_size);
    print_matrix(dmat_c, rowsa, colsb, 'C');

    if (rowsa * colsa < cpu_limit && colsa * colsb < cpu_limit) {
        validate_results(hmat_c, dmat_c, rowsa * colsb, tolerance);
    }
}
