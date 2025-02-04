#include "dragonfly-common.h"
#include "utils.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include<math.h>
#include <time.h>



float eval(float *wi, unsigned int *seed, unsigned int d) {
  for (int i = 0; i < 14; i++) {
    if (wi[i] < 0.0) {
      wi[i] = 0.0;
    }
    if (i % 2 == 0 && wi[i] == wi[i + 1]) {
      wi[i + 1] += 0.0001;
    }
  }
  (void)d;
  Weights w = {
      // exploring
      .al = {wi[0], wi[1]},
      .cl = {wi[2], wi[3]},
      // swarming
      .sl = {wi[4], wi[5]},
      .fl = {wi[6], wi[7]},
      .el = {wi[8], wi[9]},
      .wl = {wi[10], wi[11]},
      .ll = {wi[12], wi[13]},
  };
  Parameters p = {.n = 128, .dim = 10, .chunks = 64, .iterations = 200};

  Fitness fitness = shifted_fitness;
  int N = 30;
  float *array = malloc(sizeof(float)*N);
  for (int i = 0; i < N; i++) {

    float *shifted_tmp = malloc(sizeof(float) * p.dim);
    float *shifted_rotation = malloc(sizeof(float) * p.dim * p.dim);
    float *shifted_shift = init_array(p.dim, 7.0, seed);
    init_matrix(shifted_rotation, 10.0, p.dim, seed);

    init_shifted_fitness(shifted_tmp, shifted_rotation, shifted_shift,
                         rastrigin_fitness);

    float *res = dragonfly_compute(p, w, fitness, 1, 0, 100.0, *seed);
    array[i] = fitness(res, seed, p.dim);
    free(shifted_tmp);
    free(shifted_rotation);
    free(shifted_shift);
    free(res);
  }
  float std=0.0, avg=0.0;
  for(int i=0; i<N; i++){
    avg+=array[i];
  }
  avg/=N;
  for(int i=0; i<N; i++){
    std+=(array[i]-avg)*(array[i]-avg);
  }
  std=sqrt(std/N);
  printf("avg: %f\n std %f", avg, std);
  return avg;
}

int main() {
  /*
  Weights w = {
    // exploring
    .al = {0.0, 0.00},
    .cl = {0.00, 0.45},
    // swarming
    .sl = {0.0, 0.0},
    .fl = {0.7, 0.9},
    .el = {0.0, 0.0},
    .wl = {0.9, 0.4},
    .ll = {0.11, 0.3},
    .max_speedl = {1.7, 2.15
};
  */
  srand(time(NULL));
  float best[14] = {
      0.000000,
0.000100,
0.000000,
0.000100,
0.000000,
0.000100,
0.000000,
1.251300,
0.000000,
0.000100,
0.000000,
0.000100,
0.000000,
0.000100

  };
  unsigned int seed =rand();
  eval(best, &seed, 10);
  //printf("starting: %f\n", best_fitness);
  /*float cur[16];
  unsigned int seed = rand();
  while (true) {
    for (int i = 0; i < 16; i++) {
      cur[i] = best[i];
      // if(rand()%8==0){
      cur[i] = best[i] + RAND_FLOAT(0.05, &seed);
      if (cur[i] < 0.0) {
        cur[i] = 0.0;
      }
      //}
    }
    float fit = eval(cur);
    if (fit > best_fitness) {
      best_fitness = fit;
      printf("New min %f\n", best_fitness);
      for (int i = 0; i < 16; i++) {
        printf("%f, ", cur[i]);
      }
      printf("\n");
      memcpy(best, cur, sizeof(float) * 16);
    }
  }*/
}
/*
New min -353743.656250
0.181880 -0.102300 -0.035865 0.361603 0.596336 0.018159 0.651550 0.487240
-0.064944 0.030771 0.898261 0.121631 0.063970 0.295120 1.880680 2.105216

New min -1655767.000000
0.343304 0.120054 0.147655 0.456489 0.523672 0.009136 0.579197 0.649009 0.000000
0.011433 0.626118 0.226548 0.070795 0.611347 2.347073 1.677076
*/
