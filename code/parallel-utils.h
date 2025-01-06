#ifndef DA_UTILS
#define DA_UTILS

float* init_array_parallel(unsigned int dimensions, float range_max,int nr_threads);
void parallel_sum(float* cumulated_pos,float* average_speed,float* positions,float* speeds,unsigned int N,unsigned int dimensions,unsigned int tot_threads);


#endif