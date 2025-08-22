#include <mpi.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
float wi[14] = {
  0.000000,
0.000100,
1.023072,
1.628618,
0.000000,
0.000100,
0.930669,
1.020353,
0.000000,
0.100000,
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
  ChunkSize c = new_chunk_size(p.starting_chunk_count, 1, p.iterations);

  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Fitness fitness = rastrigin_fitness;

  float *res = dragonfly_compute(p, w, c, fitness, comm_size, rank, 100.0, start_time);

  MPI_Barrier(MPI_COMM_WORLD);
  unsigned int s = 0;
  float fit = fitness(res, &s, p.problem_dimensions);

  if (rank == 0) {
    printf("found fitness=%f\n", fit);
    for (unsigned int i = 0; i < p.problem_dimensions; i++) {
      printf("%f\n", res[i]);
    }
    double duration = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    printf("Execution time = %f\n", duration);
  }
  free(res);
  MPI_Finalize();
}
