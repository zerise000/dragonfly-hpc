#ifndef DA_PARALLEL
#define DA_PARALLEL
#include"dragonfly-common.h"
#include <string.h>
#include <stdlib.h>
#include"utils.h"

void dragonfly_compute_step(Dragonfly *d, float *average_speed,
                            float *cumulated_pos, float *food, float *enemy,
                            unsigned int N);
void computation_accumulate(ComputationStatus *message, Dragonfly *d, float* best, float* best_fitness);

void set_thread_number(unsigned int N);

#endif
