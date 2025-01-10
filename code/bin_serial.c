#include "dragonfly-common.h"
#include "utils.h"
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

float *dragonfly_compute(Dragonfly *d, unsigned int chunks, unsigned int dim,
                         unsigned int iter);

typedef struct{
  unsigned int n, chunks, iterations, dim;


} Parameters;

Parameters parameter_parse(int argc, char* argv[]){
  Parameters p;
  if(argc!=5){
    fprintf(stderr, "invalid parameter count: expected n, chunks, iterations, dimensions");
    exit(-1);
  }
  p.n=atoi(argv[1]);
  p.chunks=atoi(argv[2]);
  p.iterations=atoi(argv[3]);
  p.dim=atoi(argv[4]);
  if(p.n==0 || p.chunks==0 || p.iterations==0 || p.dim==0){
    fprintf(stderr, "invalid parameter they must be all bigger than 1 (and integers)");
    exit(-1);
  }
  if((p.chunks&(p.chunks-1))!=0){
    fprintf(stderr, "chunk must be a power of two");
    exit(-1);
  }
  return p;
};

int main(int argc, char* argv[]) {
  // take input
  clock_t start_time;
  start_time=clock();
  // set parameters
  Parameters p = parameter_parse(argc, argv);
  Fitness fitness = rastrigin_fitness;
  Weights w = {
      // exploring
      .al = {0.2, 0.01},
      .cl = {0.01, 0.2},
      // swarming
      .sl = {0.1, 0.1},
      .fl = {0.7, 0.7},
      .el = {0.0, 0.0},
      .wl = {0.8, 0.8},
      .ll = {0.2, 0.2},
  };
  
  Dragonfly *d = malloc(sizeof(Dragonfly) * p.chunks);

  for (unsigned int i = 0; i < p.chunks; i++) {
    srand(i);
    d[i] = dragonfly_new(p.dim, p.n, p.chunks, i, p.iterations, 5.0, w,
                         fitness);
    dragonfly_alloc(&d[i]);
  }

  float *res = dragonfly_compute(d, p.chunks, p.dim, p.iterations);

  for (unsigned int i = 0; i < p.chunks; i++) {
    dragonfly_free(d[i]);
  }
  free(d);

  float fit = fitness(res, p.dim);

  printf("found fitness=%f\n", fit);
  for (unsigned int i = 0; i < p.dim; i++) {
    printf("%f\n", res[i]);
  }
  free(res);

  double duration = (double)(clock() - start_time)/CLOCKS_PER_SEC; 
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

// take timing not including IO
float *dragonfly_compute(Dragonfly *d, unsigned int chunks, unsigned int dim,
                         unsigned int iter) {
  float *best = malloc(sizeof(float) * dim);
  memcpy(best, d[0].positions, dim * sizeof(float));
  float best_fitness = d[0].fitness(best, dim);
  printf("starting fitness %f\n", best_fitness);

  unsigned int joint_chunks = 1;
  int log_chunks = 0;
  int tmp_chunks = chunks;
  while (tmp_chunks > 1) {
    tmp_chunks /= 2;
    log_chunks++;
  }

  unsigned int update_chunk_steps = (iter + log_chunks) / (log_chunks + 1);
  Message *messages = malloc(sizeof(Message) * chunks * 2);
  raw_sendrecv_shift = chunks;
  // for each iteration
  for (unsigned int i = 0; i < iter; i++) {
    if (i != 0 && i % update_chunk_steps == 0) {
      joint_chunks *= 2;
    }
    // compute avarage speed and positions
    for (unsigned int j = 0; j < chunks; j++) {
      zeroed(messages[j].cumulated_pos, dim);
      zeroed(messages[j].cumulated_speeds, dim);
      memcpy(messages[j].next_enemy, d[j].positions, sizeof(float) * dim);
      memcpy(messages[j].next_food, d[j].positions, sizeof(float) * dim);
      messages[j].next_enemy_fitness =
          d[j].fitness(messages[j].next_enemy, dim);
      messages[j].next_food_fitness = d[j].fitness(messages[j].next_food, dim);
      messages[j].n = 0;
      for (unsigned int k = 0; k < d[j].N; k++) {
        float *cur_pos = d[j].positions + dim * k;
        sum_assign(messages[j].cumulated_pos, cur_pos, dim);
        sum_assign(messages[j].cumulated_speeds, d[j].speeds + dim * k, dim);
        float fitness = d[j].fitness(cur_pos, dim);
        if (fitness > messages[j].next_food_fitness) {
          memcpy(messages[j].next_food, cur_pos, sizeof(float) * dim);
          messages[j].next_food_fitness = fitness;
        }
        if (fitness < messages[j].next_enemy_fitness) {
          memcpy(messages[j].next_enemy, cur_pos, sizeof(float) * dim);
          messages[j].next_enemy_fitness = fitness;
        }
        if (fitness > best_fitness) {

          memcpy(best, cur_pos, sizeof(float) * dim);
          best_fitness = fitness;
        }
        messages[j].n += 1;
      }
    }

    // computed, then broadcast to others

    // execute log2(joint_chunks)
    for (unsigned int s = 1; s < joint_chunks; s *= 2) {
      memcpy(((void *)messages) + sizeof(Message) * chunks, messages,
             sizeof(Message) * chunks);
      for (unsigned int j = 0; j < chunks; j++) {

        message_broadcast(&messages[j], j, s, messages, dim, raw_sendrecv);
      }
    }

    // prepare and compute step
    for (unsigned int j = 0; j < chunks; j++) {
      scalar_prod_assign(messages[j].cumulated_speeds,
                         1.0 / (float)messages[j].n, dim);
      dragonfly_compute_step(&d[j], messages[j].cumulated_speeds,
                             messages[j].cumulated_pos, messages[j].next_food,
                             messages[j].next_enemy);
    }
  }
  free(messages);
  // check last iteration
  for (unsigned int i = 0; i < chunks; i++) {
    for (unsigned int j = 0; j < d[i].N; j++) {
      float *cur_pos = d[i].positions + j * dim;
      float fitness = d[i].fitness(cur_pos, dim);
      if (fitness > best_fitness) {
        printf("found fit %f\n", fitness);
        memcpy(best, cur_pos, sizeof(float) * dim);
        best_fitness = fitness;
      }
    }
  }
  return best;
}


