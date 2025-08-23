#ifndef DA_UTILS
#define DA_UTILS
#include "dragonfly-common.h"
#include <stdio.h>
#define M_PI 3.14159265358979323846
#include <stdlib.h> 
#define RAND_FLOAT(N, seed) (((float)rand_r(seed) / (float)RAND_MAX) * N * 2 - N)
#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

#ifdef USE_OPENMP
#include <omp.h>
#endif

float *init_array(unsigned int dimensions, float range_max, unsigned int *seed);
void sum_assign(float *dest, float *source, unsigned int size);
void sub_assign(float *dest, float *source, unsigned int size);
void scalar_prod_assign(float *dest, float val, unsigned int size);
float length(float *inp, unsigned int size);
void zeroed(float *dest, unsigned int size);

void brownian_motion(float* inp, unsigned int dim, unsigned int * seed);
// fitness functions
float rastrigin_fitness(float *inp, unsigned int *, unsigned int dim);
float sphere_fitness(float *inp, unsigned int *, unsigned int dim);
float rosenblock_fitness(float *inp, unsigned int*, unsigned int dim);

float shifted_fitness(float *inp, unsigned int* seed, unsigned int dim);
void init_shifted_fitness(float *tmp, float * rotation, float * shift, Fitness fitness);
void init_matrix(float *inp, float range_max, unsigned int dim, unsigned int *seed);


void dragonfly_compute_step(Dragonfly *d, float *average_speed,
    float *cumulated_pos, float *food, float *enemy,
    unsigned int N, unsigned int NUM_THREADS);
void new_computation_accumulate(Dragonfly *d, 
																LogicalChunk *current_chunk,
                                unsigned int *seed,
																unsigned int nr_threads);

#endif
