#include "dragonfly-common.h"
#include "utils.h"
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DA_SERIAL_LIB
#define DA_MPICH_LIB
#include "bin_serial.c"
#include "bin_mpich.c"

float eval(float *wi, unsigned int d) {
    for(int i=0; i<16; i++){
        if(wi[i]<0.0){
            wi[i]=0.0;
        }
        if(i%2==0&&wi[i]==wi[i+1]){
            wi[i+1]+=0.0001;
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
      .max_speedl = {wi[14], wi[15]},
  };
  Parameters p = {.n = 100, .dim = 10, .chunks = 8, .iterations = 1000};

  Fitness fitness = shifted_fitness;
  float avg = 0.0;
  int N = 40;
  for (int i = 0; i < N; i++) {
    //printf("run %d\n", i);
    unsigned int seed = rand();

    float *shifted_tmp = malloc(sizeof(float) * p.dim);
    float *shifted_rotation = malloc(sizeof(float) * p.dim * p.dim);
    float *shifted_shift = init_array(p.dim, 100.0, &seed);
    init_matrix(shifted_rotation, 100.0, p.dim, &seed);

    init_shifted_fitness(shifted_tmp, shifted_rotation, shifted_shift,
                         rastrigin_fitness);

    float *res = dragonfly_serial_compute(p, w, fitness, seed);
    // printf("%f\n", fitness(res, p.dim));
    avg += fitness(res, p.dim);
    free(shifted_tmp);
    free(shifted_rotation);
    free(shifted_shift);
    free(res);
  }
  // printf("avg: %f\n", avg);
  return avg / N;
}

int main() {
  MPI_Init(NULL, NULL);
  // wait for all the process to start
  MPI_Barrier(MPI_COMM_WORLD);
  clock_t start_time;
  start_time=clock();
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  srand(time(NULL) + rank);
  Fitness fitness = eval;
  Parameters p = {.n = 8, .dim = 16, .chunks = 8, .iterations = 2};
  /*float best[16] = {
      0.0,      0.05, 0.12, 0.33, 0.000000, 0.04, 0.85, 0.95,
      0.000000, 0.0,  0.86, 0.41, 0.05,     0.19, 2.4,  1.4,
  };
  float best_fitness = fitness(best, 0);*/

  if ((int)p.chunks != comm_size) {
    fprintf(stderr, "chunks!=comm_size (%d!=%d)", p.chunks, comm_size);
  }

  float wi[16] = {0.017683, 0.000000, 0.020466, 0.230936, 0.023082, 0.001584,
                  0.950145, 0.919330, 0.001764, 0.000000, 0.882060, 0.468423,
                  0.089438, 0.224199, 0.2336837, 0.1365714};
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
      .max_speedl = {wi[14], wi[15]},
  };
  Dragonfly d = dragonfly_new(p.dim, p.n, p.chunks, rank, p.iterations, 5.0, w,
                              fitness, rank);
  dragonfly_alloc(&d);

  float *res = dragonfly_compute(&d, p.chunks, p.dim, p.iterations);
  dragonfly_free(d);

  MPI_Barrier(MPI_COMM_WORLD);

  

  if (rank == 0) {
    printf("end, evaluating");
    float fit = fitness(res, p.dim);
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