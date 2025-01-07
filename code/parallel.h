#ifndef __PARALLEL_H__
#define __PARALLEL_H__

#include "wrappers.h"
#include "utils.h"

void update(Entity* food,Entity* enemy,Temp temp,Dragonfly dragonflies,Weights weights,float space_size,unsigned int dimensions,unsigned int nr_threads, float (*fitness)(float*, unsigned int));
void parallel_sum(Temp* temp,Dragonfly dragonflies,unsigned int N,unsigned int dimensions,unsigned int tot_threads);
float* init_array_parallel(unsigned int dimensions, float range_max,int nr_threads);

#endif
