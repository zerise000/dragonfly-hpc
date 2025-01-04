#include <stdlib.h>
#include <math.h>
#include "utils.h"

float* init_array(unsigned int dimensions, float range_max){
	float *arr = malloc(dimensions*sizeof(float));
	for(unsigned int i=0; i<dimensions; i++){
      arr[i] = RAND_FLOAT(range_max);
	}
	return arr;
}

void zeroed(float* dest, unsigned int size){
	for(unsigned int i=0; i<size; i++){
		dest[i]=0.0;
	}
}

void sum_assign(float* dest, float* source, unsigned int size){
	for(unsigned int i=0; i<size; i++){
		dest[i]+=source[i];
	}
}

void sub_assign(float* dest, float* source, unsigned int size){
	for(unsigned int i=0; i<size; i++){
		dest[i]-=source[i];
	}
}

void scalar_prod_assign(float* dest, float val, unsigned int size){
	for(unsigned int i=0; i<size; i++){
		dest[i]*=val;
	}
}

float length(float* inp, unsigned int size){
	float ret=0.0;
	for(unsigned int i=0; i<size; i++){
		ret+=inp[i]*inp[i];
	}
	return sqrt(ret);
}
float sigma(float beta) {
  return tgamma(beta+1)*sin(beta*M_PI/2)/tgamma((beta+1)/2)*beta*pow(2,(beta-1)/2);
}
float levy(){
  float beta = 4.25;
  return 0.01*(RAND_FLOAT(1)*sigma(beta)/pow(RAND_FLOAT(1),1/beta));
}