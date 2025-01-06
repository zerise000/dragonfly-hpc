#ifndef __PARALLEL_H__
#define __PARALLEL_H__


void parallel_sum(float* cumulated_pos,float* average_speed,float* positions,float* speeds,unsigned int N,unsigned int dimensions,unsigned int tot_threads);
float* init_array_parallel(unsigned int dimensions, float range_max,int nr_threads);

#endif
