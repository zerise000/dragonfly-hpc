#include "dragonfly-common.h"
#include"utils-special.h"
#include <stdio.h>

void dragonfly_compute_step(Dragonfly *d, float *average_speed,
                            float *cumulated_pos, float * food, float * enemy, unsigned int N) {
  unsigned int dim = d->dim;
  // for each dragonfly
  for (unsigned int j = 0; j < d->N; j++) {
    float *cur_pos = d->positions + dim * j;
    float *cur_speed = d->speeds + dim * j;

    // compute separation: Si = -sumall(X-Xi)
    memcpy(d->S, cumulated_pos, sizeof(float) * dim);
    scalar_prod_assign(d->S, 1.0/(float)N, dim);
    sub_assign(d->S, cur_pos, dim);
    scalar_prod_assign(d->S, d->w.s, dim);

    // compute alignament: Ai = avarage(Vi)
    memcpy(d->A, average_speed, sizeof(float) * dim);
    scalar_prod_assign(d->A, d->w.a, dim);

    // compute cohesion: Ci = avarage(Xi)-X
    memcpy(d->C, cumulated_pos, sizeof(float) * dim);
    scalar_prod_assign(d->C, 1.0 / (float)N, dim);
    sub_assign(d->C, cur_pos, dim);
    scalar_prod_assign(d->C, d->w.c, dim);

    // food attraction: Fi=X_food - X
    memcpy(d->F, food, sizeof(float) * dim);
    sub_assign(d->F, cur_pos, dim);
    scalar_prod_assign(d->F, d->w.f, dim);

    // enemy repulsion: E=X_enemy+X
    memcpy(d->E, enemy, sizeof(float) * dim);
    sum_assign(d->E, cur_pos, dim);
    scalar_prod_assign(d->E, d->w.e, dim);

    brownian_motion(d->levy, dim, &d->seed);
    

    // compute speed = sSi + aAi + cCi + fFi + eEi + w
    scalar_prod_assign(cur_speed, d->w.w, dim);
    sum_assign(cur_speed, d->E, dim);
    sum_assign(cur_speed, d->F, dim);
    sum_assign(cur_speed, d->C, dim);
    sum_assign(cur_speed, d->A, dim);
    sum_assign(cur_speed, d->S, dim);
    sum_assign(cur_speed, d->levy, dim);
    
    
    // update current pos
    sum_assign(cur_pos, cur_speed, dim);
  }

  // update weights
  weights_step(&d->w);
}

void computation_accumulate(ComputationStatus *message, Dragonfly *d, float* best, float* best_fitness){
  unsigned int dim=d->dim;
  zeroed(message->cumulated_pos, dim);
  zeroed(message->cumulated_speeds, dim);
  memcpy(message->next_enemy, d->positions, sizeof(float) * dim);
  memcpy(message->next_food, d->positions, sizeof(float) * dim);
  message->next_enemy_fitness =
      d->fitness(message->next_enemy, dim);
  message->next_food_fitness = d->fitness(message->next_food, dim);
  message->n = 0;
  for (unsigned int k = 0; k < d->N; k++) {
    float *cur_pos = d->positions + dim * k;
    sum_assign(message->cumulated_pos, cur_pos, dim);
    sum_assign(message->cumulated_speeds, d->speeds + dim * k, dim);
    float fitness = d->fitness(cur_pos, dim);
    if (fitness > message->next_food_fitness) {
      memcpy(message->next_food, cur_pos, sizeof(float) * dim);
      message->next_food_fitness = fitness;
    }
    if (fitness < message->next_enemy_fitness) {
      memcpy(message->next_enemy, cur_pos, sizeof(float) * dim);
      message->next_enemy_fitness = fitness;
    }
    if (fitness > *best_fitness) {
      memcpy(best, cur_pos, sizeof(float) * dim);
      *best_fitness = fitness;
    }
    message->n += 1;
  }
}
