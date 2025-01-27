#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "utils.h"
#include "dragonfly-common.h"
#include "utils-special.h"

float *dragonfly_serial_compute(Parameters p, Weights w, Fitness f,
                                unsigned int srand);

#ifndef DA_SERIAL_LIB
int main(int argc, char *argv[]) {
  // start clock
  clock_t start_time;
  start_time = clock();

  // set parameters
  Parameters p = parameter_parse(argc, argv);
  unsigned int seed = time(NULL);
  float *shifted_tmp = malloc(sizeof(float) * p.dim);
  float *shifted_rotation = malloc(sizeof(float) * p.dim * p.dim);
  float *shifted_shift = init_array(p.dim, 100.0, &seed);
  init_matrix(shifted_rotation, 100.0, p.dim, &seed);

  init_shifted_fitness(shifted_tmp, shifted_rotation, shifted_shift,
                       rastrigin_fitness);

  Fitness fitness = shifted_fitness;

  float wi[14] = {0.941010, 0.000000, 0.000000, 1.347089, 0.063430, 0.000000,
                  3.548271, 2.154025, 0.000000, 0.000100, 2.139098, 3.452764,
                  0.707045, 3.671526

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

  float *res = dragonfly_serial_compute(p, w, fitness, seed);

  float fit = fitness(res, p.dim);

  printf("found fitness=%f\n", fit);
  for (unsigned int i = 0; i < p.dim; i++) {
    printf("%f\n", res[i]);
  }
  free(res);
  free(shifted_rotation);
  free(shifted_shift);
  free(shifted_tmp);
  double duration = (double)(clock() - start_time) / CLOCKS_PER_SEC;
  printf("Execution time = %f\n", duration);
}
#endif

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
float *dragonfly_serial_compute(Parameters p, Weights w, Fitness fitness,
                                unsigned int srand) {
  if (p.chunks==0){
    fprintf(stderr, "chunks==0");
    exit(-2);
  }
  unsigned int dim = p.dim;
  unsigned int chunks = p.chunks;

  Dragonfly *d = malloc(sizeof(Dragonfly) * p.chunks);
  // allocate problem
  for (unsigned int i = 0; i < p.chunks; i++) {
    d[i] = dragonfly_new(p.dim, p.n, p.chunks, i, p.iterations, 100.0, w,
                         fitness, i + srand);
    dragonfly_alloc(&d[i]);
  }

  float *best = malloc(sizeof(float) * dim);
  memcpy(best, d->positions, dim * sizeof(float));
  float best_fitness = d->fitness(best, dim);
  // printf("starting fitness %f\n", best_fitness);

  unsigned int joint_chunks = 1;
  int log_chunks = 0;
  int tmp_chunks = chunks;
  while (tmp_chunks > 1) {
    tmp_chunks /= 2;
    log_chunks++;
  }

  unsigned int update_chunk_steps =
      (p.iterations + log_chunks) / (log_chunks + 1);
  Message *messages = malloc(sizeof(Message) * chunks * 2);
  raw_sendrecv_shift = chunks;
  // for each iteration
  for (unsigned int i = 0; i < p.iterations; i++) {
    if (i != 0 && i % update_chunk_steps == 0) {
      joint_chunks *= 2;
    }
    // compute avarage speed and positions
    for (unsigned int j = 0; j < chunks; j++) {
      message_acumulate(&messages[j], &d[j], best, &best_fitness);
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
                             messages[j].next_enemy, messages[j].n);
    }
  }

  // check last iteration
  for (unsigned int i = 0; i < chunks; i++) {
    message_acumulate(&messages[i], &d[i], best, &best_fitness);
  }
  if (best_fitness > messages[0].next_food_fitness) {
    memcpy(messages[0].next_food, best, dim * sizeof(float));
    messages[0].next_food_fitness = best_fitness;
  }
  for (unsigned int i = 0; i < chunks; i++) {
    for (unsigned int s = 1; s < chunks; s *= 2) {
      memcpy(((void *)messages) + sizeof(Message) * chunks, messages,
             sizeof(Message) * chunks);
      for (unsigned int j = 0; j < chunks; j++) {
        message_broadcast(&messages[j], j, s, messages, dim, raw_sendrecv);
      }
    }
  }
  memcpy(best, messages[0].next_food, dim * sizeof(float));
  free(messages);
  // free d

  for (unsigned int i = 0; i < p.chunks; i++) {
    dragonfly_free(d[i]);
  }
  free(d);

  return best;
}
