#include "dragonfly-common.h"
#include "utils.h"
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
  float avg = 0.0;
  int N = 30;
  for (int i = 0; i < N; i++) {

    float *shifted_tmp = malloc(sizeof(float) * p.dim);
    float *shifted_rotation = malloc(sizeof(float) * p.dim * p.dim);
    float *shifted_shift = init_array(p.dim, 7.0, seed);
    init_matrix(shifted_rotation, 10.0, p.dim, seed);

    init_shifted_fitness(shifted_tmp, shifted_rotation, shifted_shift,
                         rastrigin_fitness);

    float *res = dragonfly_compute(p, w, fitness, 1, 0, 100.0, *seed);
    avg += fitness(res, seed, p.dim);
    free(shifted_tmp);
    free(shifted_rotation);
    free(shifted_shift);
    free(res);
  }

  return avg / N;
}

int main(int argc, char **argv) {
  MPI_Init(NULL, NULL);
  // wait for all the process to start
  MPI_Barrier(MPI_COMM_WORLD);
  clock_t start_time;
  start_time = clock();
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  srand(time(NULL) + rank);
  Fitness fitness = eval;
  Parameters p = parameter_parse(argc, argv);

  float wi[14] = {
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
  float *res = dragonfly_compute(p, w, fitness, comm_size, rank, 2.0, 0);

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    unsigned int seed = 0;
    float fit = fitness(res, &seed, p.dim);
    printf("found fitness=%f\n", fit);
    for (unsigned int i = 0; i < p.dim; i++) {
      printf("%f\n", res[i]);
    }
    double duration = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    printf("Execution time = %f\n", duration);
  }
  free(res);
  MPI_Finalize();
}