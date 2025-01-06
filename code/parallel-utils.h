#ifndef DA_PARALLEL
#define DA_PARALLEL

float* init_array_parallel(unsigned int dimensions, float range_max,int nr_threads);
void parallel_sum(float* cumulated_pos,float* average_speed,float* positions,float* speeds,unsigned int N,unsigned int dimensions,unsigned int tot_threads);


#endif