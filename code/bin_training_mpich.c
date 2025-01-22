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
  int N = 50;
  for (int i = 0; i < N; i++) {
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
  Parameters p = {.n = 100, .dim = 16, .chunks = 2, .iterations = 1000};
  float best[16] = {
      0.0,      0.05, 0.12, 0.33, 0.000000, 0.04, 0.85, 0.95,
      0.000000, 0.0,  0.86, 0.41, 0.05,     0.19, 2.4,  1.4,
  };
  float best_fitness = fitness(best, 0);

  if ((int)p.chunks != comm_size) {
    fprintf(stderr, "chunks!=comm_size (%d!=%d)", p.chunks, comm_size);
  }

  if (rank == 0) {
    printf("starting: %f\n", best_fitness);
  }

  float wi[16] = {0.017683, 0.000000, 0.020466, 0.230936, 0.023082, 0.001584,
                  0.950145, 0.919330, 0.001764, 0.000000, 0.882060, 0.468423,
                  0.089438, 0.224199, 2.336837, 1.365714};
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

  float fit = fitness(res, p.dim);

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

  /*float cur[16];


  unsigned int seed = rand();
  while(true){
      for(int i=0; i<16; i++){
          cur[i]=best[i];
          //if(rand()%8==0){
              cur[i]=best[i]+RAND_FLOAT(0.05, &seed);
              if(cur[i]<0.0){
                  cur[i]=0.0;
              }
          //}

      }
      float fit = eval(cur);
      if(fit>best_fitness){
          best_fitness=fit;
          printf("New min %f\n", best_fitness);
          for(int i=0; i<16; i++){
              printf("%f, ", cur[i]);
          }
          printf("\n");
          memcpy(best, cur, sizeof(float)*16);
      }
  }*/
}

/*


    // start clock
  clock_t start_time;
  start_time=clock();

  // set parameters
  Parameters p = parameter_parse(argc, argv);

  Weights w = {
      // exploring
      .al = {0.3, 0.00},
      .cl = {0.00, 0.3},
      // swarming
      .sl = {0.4, 0.0},
      .fl = {0.7, 0.7},
      .el = {0.0, 0.0},
      .wl = {0.8, 0.8},
      .ll = {0.2, 0.3},
      .max_speedl = {0.1, 0.03},
  };

  Fitness fitness = rastrigin_fitness;


  Dragonfly d = dragonfly_new(p.dim, p.n, p.chunks,  rank, p.iterations, 5.0, w,
                         fitness, rank);
  dragonfly_alloc(&d);

  float *res = dragonfly_compute(&d, p.chunks, p.dim, p.iterations);
  dragonfly_free(d);

  MPI_Barrier(MPI_COMM_WORLD);

  float fit = fitness(res, p.dim);

  if(rank==0){
    printf("found fitness=%f\n", fit);
    for (unsigned int i = 0; i < p.dim; i++) {
      printf("%f\n", res[i]);
    }
    double duration = (double)(clock() - start_time)/CLOCKS_PER_SEC;
    printf("Execution time = %f\n", duration);
  }
  free(res);
  MPI_Finalize();
}
*/

/*
New min -353743.656250
0.181880 -0.102300 -0.035865 0.361603 0.596336 0.018159 0.651550 0.487240
-0.064944 0.030771 0.898261 0.121631 0.063970 0.295120 1.880680 2.105216

New min -1655767.000000
0.343304 0.120054 0.147655 0.456489 0.523672 0.009136 0.579197 0.649009 0.000000
0.011433 0.626118 0.226548 0.070795 0.611347 2.347073 1.677076
*/
