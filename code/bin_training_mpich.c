#include "dragonfly-common.h"
#include "utils.h"
#include <mpi.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

float eval(float *wi, unsigned int *seedi, unsigned int d, void *data) {
  (void)data;
  if (d < 14) {
    fprintf(stderr,
            "impossible to compute with %d dimensions it must be at least 14\n",
            d);
    exit(-1);
  }
  unsigned int seed = *seedi;
  for (int i = 0; i < 14; i++) {
    if (wi[i] < 0.0) {
      wi[i] = 0.0;
    }
    if (wi[i] > 4.0) {
      wi[i] = 4.0;
    }
    if (i % 2 == 0 && wi[i] == wi[i + 1]) {
      wi[i + 1] += 0.0001;
    }
  }
  float m = max(wi[8], wi[9]);
  if (m > 0.1) {
    wi[8] /= 10 * m;
    wi[9] /= 10 * m;
  }
  m = max(wi[10], wi[11]);
  if (m > 1.0) {
    wi[10] /= 1.2 * m;
    wi[11] /= 1.2 * m;
  }

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
  Parameters p = {.population_size = 128,
                  .problem_dimensions = 10,
                  .starting_chunk_count = 64,
                  .iterations = 80,
                  .threads_per_process = 1};
  ChunkSize c = new_chunk_size(p.starting_chunk_count, 1, p.iterations);
  Fitness fitness = rastrigin_fitness;
  float avg = 0.0;
  int N = 30;
  for (int i = 0; i < N; i++) {

    void *shifted_fitness_data = malloc_shifted_fitness(fitness, 80.0, 90.0, &seed,
                                                   p.problem_dimensions);

    float *res =
        dragonfly_compute(p, w, c, shifted_fitness, shifted_fitness_data,
                          sizeof(ShiftedFitnessParams), 1, 0, 100.0, seed);

    avg += fitness(res, &seed, p.problem_dimensions, shifted_fitness_data);
    /*printf("%f ", fitness(res, seed, p.dim));
    for(int i=0; i<14; i++){
      printf("%2f ", wi[i]);
    }
    printf("\n");*/
    free_shifted_fitness(shifted_fitness_data);
    free(res);
  }
  return avg / N;
}

int main(int argc, char **argv) {
  MPI_Init(NULL, NULL);
  // wait for all the process to start
  MPI_Barrier(MPI_COMM_WORLD);
  struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  srand(time(NULL) + rank);
  Fitness fitness = eval;
  Parameters p = parameter_parse(argc, argv);
  ChunkSize c = new_chunk_size(p.starting_chunk_count, 1, p.iterations);
  float wi[14] = {0.000000, 0.000100, 1.023072, 1.628618, 0.000000,
                  0.000100, 0.930669, 1.020353, 0.000000, 0.100000,
                  0.000000, 0.000100, 0.000000, 0.000100};
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
  
  float *res = dragonfly_compute(p, w, c, fitness, NULL, 0, comm_size, rank,
                                 2.0, start_time.tv_nsec + rank);

  // float *res = malloc(sizeof(float)*p.population_size);
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    unsigned int seed = 0;
    float fit = fitness(res, &seed, p.problem_dimensions, NULL);
    printf("found fitness=%f\n", fit);
    for (unsigned int i = 0; i < p.problem_dimensions; i++) {
      printf("%f\n", res[i]);
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double duration = (double)(end_time.tv_nsec - start_time.tv_nsec) / 1e9 + (end_time.tv_sec - start_time.tv_sec);
    printf("Execution time = %f\n", duration);
  }
  free(res);
  MPI_Finalize();
}