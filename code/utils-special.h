#ifndef DA_PARALLEL
#define DA_PARALLEL
#include"dragonfly-common.h"
float *init_array(unsigned int dimensions, float range_max, unsigned int *seed);

void dragonfly_compute_step(Dragonfly *d, float *average_speed,
                            float *cumulated_pos, float *food, float *enemy,
                            unsigned int N);
void message_acumulate(Message *message, Dragonfly *d, float* best, float* best_fitness);

#endif