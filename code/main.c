#include "dragonfly-common.h"
#include "utils.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// take timing not including IO
float *dragonfly_compute(Dragonfly *d) {
  unsigned int dim = d->dim;
  // for each iteration
  for (unsigned int i = 0; i < d->iter; i++) {
    // compute avarage speed and positions
    zeroed(d->cumulated_pos, dim);
    zeroed(d->average_speed, dim);
    for (unsigned int j = 0; j < d->N; j++) {
      sum_assign(d->cumulated_pos, d->positions + dim * j, dim);
      sum_assign(d->average_speed, d->speeds + dim * j, dim);
    }
    scalar_prod_assign(d->average_speed, 1.0 / (float)d->N, dim);
    dragonfly_compute_step(d, d->average_speed, d->cumulated_pos);
  }
  return d->next_food;
}

int main() {
  srand(time(NULL));
  Weights w = {
      // exploring
      .al = {0.3, 0.01},
      .cl = {0.01, 0.3},
      // swarming
      .sl = {0.1, 0.1},
      .fl = {0.1, 0.1},
      .el = {0.1, 0.1},
      .wl = {0.1, 0.1},
  };
  unsigned int dim = 2;
  Dragonfly d = dragonfly_new(dim, 500, 500, 5.0, w, rosenblock_fitness);
  dragonfly_alloc(&d);
  float *tmp = dragonfly_compute(&d);
  float res[20];
  memcpy(res, tmp, sizeof(float) * dim);
  dragonfly_free(d);
  // float* res = dragonfly(dim, 500, 500, 5.0, w, rosenblock_fitness);
  float fit = sphere_fitness(res, dim);

  printf("found fitness=%f\n", fit);
  for (unsigned int i = 0; i < dim; i++) {
    printf("%f\n", res[i]);
  }
}
