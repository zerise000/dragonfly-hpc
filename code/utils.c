#include <math.h>
#include <stdlib.h>
#include "utils.h"
#include <string.h>

float *init_array(unsigned int dimensions, float range_max, unsigned int *seed) {
  unsigned int s=2;
  rand_r(&s);
  float *arr = malloc(dimensions * sizeof(float));
  for (unsigned int i = 0; i < dimensions; i++) {
    arr[i] = RAND_FLOAT(range_max, seed);
  }
  return arr;
}

void zeroed(float *dest, unsigned int size) {
  for (unsigned int i = 0; i < size; i++) {
    dest[i] = 0.0;
  }
}

void sum_assign(float *dest, float *source, unsigned int size) {
  for (unsigned int i = 0; i < size; i++) {
    dest[i] += source[i];
  }
}

void sub_assign(float *dest, float *source, unsigned int size) {
  for (unsigned int i = 0; i < size; i++) {
    dest[i] -= source[i];
  }
}

void scalar_prod_assign(float *dest, float val, unsigned int size) {
  for (unsigned int i = 0; i < size; i++) {
    dest[i] *= val;
  }
}

float dot_product(float * a, float * b, unsigned int size){
  float ret =0.0;
  for (unsigned int i = 0; i < size; i++) {
    ret += a[i] * b[i];
  }
  return ret;
}

float length(float *inp, unsigned int size) {
  float ret = dot_product(inp, inp, size);
  
  return sqrt(ret);
}

//Gram-Schmidt orthonormal matrix
void init_matrix(float *inp, float range_max, unsigned int dim, unsigned int *seed){
  float * tmp = malloc(sizeof(float)*dim);
  for(unsigned int row=0; row<dim; row++){
    for (unsigned int col = 0; col < dim; col++) {
      inp[row*dim+col] = RAND_FLOAT(range_max, seed);
    }

    //make it perpendicular
    for(unsigned int prow=0; prow<row;prow++){
      float cur=dot_product(inp+ row*dim, inp+prow*dim, dim);
      cur/=dot_product(inp+ row*dim, inp+prow*dim, dim);
      memcpy(tmp, inp+prow*dim, dim*sizeof(float));
      scalar_prod_assign(tmp, cur, dim);
      sub_assign(inp+row*dim, tmp, dim);
    }

    //normalize
    float l = length(inp+row*dim, dim);
    scalar_prod_assign(inp+row*dim, 1.0/l, dim);
  }
  free(tmp);
}

void matrix_times_vector(float* out, float *matrix, float *inp, unsigned int dim){
  for(unsigned int row=0; row<dim; row++){
    out[row]=dot_product(matrix+row*dim, inp, dim);
  }
}

/*
float sigma(float beta) {
  return tgamma(beta + 1) * sin(beta * M_PI / 2) / tgamma((beta + 1) / 2) *
         beta * pow(2, (beta - 1) / 2);
}

float levy() {
  float beta = 4.25;
  return 0.01 * (RAND_FLOAT(1) * sigma(beta) / pow(RAND_FLOAT(1), 1 / beta));
}*/

void brownian_motion(float* inp, unsigned int dim, unsigned int * seed){
  
  for(unsigned int i=0; i<dim; i++){
    inp[i]=RAND_FLOAT(1.0, seed);
  }

}

float rastrigin_fitness(float *inp, unsigned int dim) {
  float ret = 10.0 * dim;
  for (unsigned int i = 0; i < dim; i++) {
    ret += inp[i] * inp[i] - 10.0 * cos(2 * M_PI * inp[i]);
  }
  return -ret;
}

float sphere_fitness(float *inp, unsigned int dim) {
  float res = 0.0;
  for (unsigned int i = 0; i < dim; i++) {
    res += inp[i] * inp[i];
  }
  return -res;
}
float rosenblock_fitness(float *inp, unsigned int dim) {
  float res = 0.0;
  for (unsigned int i = 0; i < dim - 1; i++) {
    res +=
        100 * pow(inp[i + 1] - inp[i] * inp[i], 2.0) + pow(1.0 - inp[i], 2.0);
  }
  return -res;
}


float *ROTATION;
float *SHIFT;
float (*FITNESS)(float*, unsigned int);
float *TMP;
void init_shifted_fitness(float *tmp, float * rotation, float * shift, float (*fitness)(float*, unsigned int)){
  ROTATION = rotation;
  SHIFT = shift;
  FITNESS = fitness;
  TMP = tmp;
}


float shifted_fitness(float *inp, unsigned int dim){
  matrix_times_vector(TMP, ROTATION, inp, dim);
  sum_assign(TMP, SHIFT, dim);
  return FITNESS(TMP, dim);
}