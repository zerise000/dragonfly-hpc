#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dragonfly-common.h"
#include "utils.h"

int main(int argc, char *argv[]) {
  // start clock
  clock_t start_time;
  start_time = clock();

  // set parameters
  Parameters p = parameter_parse(argc, argv);
  unsigned int seed = time(NULL);
  float *shifted_tmp = malloc(sizeof(float) * p.problem_dimensions);
  float *shifted_rotation =
      malloc(sizeof(float) * p.problem_dimensions * p.problem_dimensions);
  float *shifted_shift = init_array(p.problem_dimensions, 100.0, &seed);
  init_matrix(shifted_rotation, 100.0, p.problem_dimensions, &seed);

  init_shifted_fitness(shifted_tmp, shifted_rotation, shifted_shift,
                       rastrigin_fitness);

  Fitness fitness = shifted_fitness;
  float wi[14] = {0.803008, 0.801540, 0.000000, 4.131542, 1.180980,
                  2.213441, 1.289022, 0.828981, 0.100000, 0.022841,
                  0.000000, 0.000000, 0.108948, 0.404676};
  Weights w = {
      // exploring
      .al = {wi[0], wi[1]},
      .cl = {wi[2], wi[3]},
      // swarming
      //.sl = {wi[4], wi[5]},
      .fl = {wi[6], wi[7]},
      .el = {wi[8], wi[9]},
      .wl = {wi[10], wi[11]},
      .ll = {wi[12], wi[13]},
  };

  float *res = dragonfly_compute(p, w, fitness, 1, 0, 100.0, seed);
  unsigned int s = 0;
  float fit = fitness(res, &s, p.problem_dimensions);

  printf("found fitness=%f\n", fit);
  for (unsigned int i = 0; i < p.problem_dimensions; i++) {
    printf("%f\n", res[i]);
  }
  free(res);
  free(shifted_rotation);
  free(shifted_shift);
  free(shifted_tmp);
  double duration = (double)(clock() - start_time) / CLOCKS_PER_SEC;
  printf("Execution time = %f\n", duration);
}

unsigned int raw_sendrecv_shift;
void raw_sendrecv(Message *send, unsigned int destination, Message *recv_buffer,
                  unsigned int source, void *data_raw) {
  Message *data = data_raw;
  (void)send;
  (void)destination;
  // send should not be needed
  *recv_buffer = data[source + raw_sendrecv_shift];
}