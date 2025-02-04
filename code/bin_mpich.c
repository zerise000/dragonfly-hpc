#include <mpi.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>
#include <time.h>
#include <unistd.h>

#include "dragonfly-common.h"
#include "utils.h"


int main(int argc, char *argv[]) {
  MPI_Init(NULL, NULL);
  // wait for all the process to start
  MPI_Barrier(MPI_COMM_WORLD);

  // start clock
  clock_t start_time;
  start_time = clock();

  // set parameters
  Parameters p = parameter_parse(argc, argv);

  Weights w = {
      // exploring
      .al = {0.1, 0.1},
      .cl = {0.7, 0.7},
      // swarming
      .sl = {0.1, 0.1},
      .fl = {1.0, 1.0},
      .el = {0.0, 0.0},
      .wl = {0.6, 0.6},
      .ll = {0.1, 0.1},
  };
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Fitness fitness = rastrigin_fitness;
  float *res = dragonfly_compute(p, w, fitness, comm_size, rank, 100.0, 0);

  MPI_Barrier(MPI_COMM_WORLD);
  unsigned int s =0;
  float fit = fitness(res, &s, p.dim);

  if (rank == 0) {
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
