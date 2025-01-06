#ifndef DA_UTILS
#define DA_UTILS

#define RAND_FLOAT(N) (((float)rand()/(float)RAND_MAX)*N*2-N)

float* init_array(unsigned int size,  float range_max);
float* init_array_parallel(unsigned int size,  float range_max,int nr_threads);
void sum_assign(float* dest, float* source, unsigned int size);
void sub_assign(float* dest, float* source, unsigned int size);
void scalar_prod_assign(float* dest, float val, unsigned int size);
float length(float* inp, unsigned int size);
void zeroed(float* dest, unsigned int size);

#endif
