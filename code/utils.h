#ifndef DA_UTILS
#define DA_UTILS
#define M_PI 3.14159265358979323846
#define RAND_FLOAT(N) (((float)rand()/(float)RAND_MAX)*N*2-N)

float* init_array(unsigned int size,  float range_max);
void sum_assign(float* dest, float* source, unsigned int size);
void sub_assign(float* dest, float* source, unsigned int size);
void scalar_prod_assign(float* dest, float val, unsigned int size);
float length(float* inp, unsigned int size);
void zeroed(float* dest, unsigned int size);

#endif