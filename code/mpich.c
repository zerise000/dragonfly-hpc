#include "dragonfly-common.h"
#include "utils.h"
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void Build_mpi_type_testsend(MPI_Datatype *data) {
  int blocklengths[7] = {50, 50, 50, 50, 1, 1, 1};
  MPI_Datatype types[7] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT,   MPI_FLOAT,
                           MPI_FLOAT, MPI_FLOAT, MPI_UNSIGNED};
  MPI_Aint displacements[7];
  displacements[0] = offsetof(Message, cumulated_pos);
  displacements[1] = offsetof(Message, cumulated_speeds);
  displacements[2] = offsetof(Message, next_enemy);
  displacements[3] = offsetof(Message, next_food);
  displacements[4] = offsetof(Message, next_enemy_fitness);
  displacements[5] = offsetof(Message, next_food_fitness);
  displacements[6] = offsetof(Message, n);
  MPI_Type_create_struct(7, blocklengths, displacements, types, data);
  MPI_Type_commit(data);
}

void raw_sendrecv(Message *send, unsigned int destination, Message *recv_buffer,
                  unsigned int source, void *data_raw) {
  MPI_Datatype *data_type = data_raw;
  MPI_Sendrecv(send, 1, *data_type, destination, 0, recv_buffer, 1, *data_type,
               source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
/*
unsigned int raw_sendrecv_shift;

void raw_sendrecv(Message *send, unsigned int destination, Message *recv_buffer,
                  unsigned int source, void *data_raw) {
  Message *data = data_raw;
  (void)send;
  (void)destination;
  // recv should not be needed
  *recv_buffer= data[source+raw_sendrecv_shift];
}*/

// take timing not including IO
float *dragonfly_compute(Dragonfly *d, unsigned int chunks, unsigned int dim,
                         unsigned int iter, unsigned int chunk_id) {

  MPI_Datatype message_type;
  Build_mpi_type_testsend(&message_type);
  float *best = malloc(sizeof(float) * dim);
  memcpy(best, d->positions, dim * sizeof(float));
  float best_fitness = d->fitness(best, dim);
  //printf("starting fitness %f\n", best_fitness);

  unsigned int joint_chunks = 1;
  int log_chunks = 0;
  int tmp_chunks = chunks;
  while (tmp_chunks > 1) {
    tmp_chunks /= 2;
    log_chunks++;
  }

  unsigned int update_chunk_steps = (iter + log_chunks) / (log_chunks + 1);
  Message message;
  //Message *messages = malloc(sizeof(Message) * chunks);
  // for each iteration
  for (unsigned int i = 0; i < iter; i++) {
    if (i != 0 && i % update_chunk_steps == 0) {
      joint_chunks *= 2;
    }
    // compute avarage speed and positions
    zeroed(message.cumulated_pos, dim);
    zeroed(message.cumulated_speeds, dim);
    memcpy(message.next_enemy, d->positions, sizeof(float) * dim);
    memcpy(message.next_food, d->positions, sizeof(float) * dim);
    message.next_enemy_fitness =
        d->fitness(message.next_enemy, dim);
    message.next_food_fitness = d->fitness(message.next_food, dim);
    message.n = 0;
    for (unsigned int k = 0; k < d->N; k++) {
      float *cur_pos = d->positions + dim * k;
      sum_assign(message.cumulated_pos, cur_pos, dim);
      sum_assign(message.cumulated_speeds, d->speeds + dim * k, dim);
      float fitness = d->fitness(cur_pos, dim);
      if (fitness > message.next_food_fitness) {
        memcpy(message.next_food, cur_pos, sizeof(float) * dim);
        message.next_food_fitness = fitness;
      }
      if (fitness < message.next_enemy_fitness) {
        memcpy(message.next_enemy, cur_pos, sizeof(float) * dim);
        message.next_enemy_fitness = fitness;
      }
      if (fitness > best_fitness) {

        memcpy(best, cur_pos, sizeof(float) * dim);
        best_fitness = fitness;
      }
      message.n += 1;
    }
    

    // computed, then broadcast to others

    // execute log2(joint_chunks)
    for (unsigned int s = 1; s < joint_chunks; s *= 2) {
      message_broadcast(&message, chunk_id, s, &message_type, dim, raw_sendrecv);
    }

    // prepare and compute step
    scalar_prod_assign(message.cumulated_speeds,
                         1.0 / (float)message.n, dim);
    dragonfly_compute_step(d, message.cumulated_speeds,
                            message.cumulated_pos, message.next_food,
                            message.next_enemy);
  }
  // check last iteration

  for (unsigned int j = 0; j < d->N; j++) {
    float *cur_pos = d->positions + j * dim;
    float fitness = d->fitness(cur_pos, dim);
    if (fitness > best_fitness) {
      printf("found fit %f\n", fitness);
      memcpy(best, cur_pos, sizeof(float) * dim);
      best_fitness = fitness;
    }
  }
  MPI_Type_free(&message_type);
  return best;
}

int main() {

  //srand(time(NULL));
  
  Weights w = {
      // exploring
      .al = {0.2, 0.01},
      .cl = {0.01, 0.2},
      // swarming
      .sl = {0.1, 0.1},
      .fl = {0.7, 0.7},
      .el = {0.0, 0.0},
      .wl = {0.8, 0.8},
  };
  unsigned int dim = 2;
  int chunks = 1;
  unsigned int iterations = 1000;
  unsigned int N = 100000;
  int my_id;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &chunks);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
  srand(my_id);
  Dragonfly d = dragonfly_new(dim, N, chunks,  my_id, iterations, 5.0, w,
                         rosenblock_fitness);
  dragonfly_alloc(&d);

  float *res = dragonfly_compute(&d, chunks, dim, iterations, my_id);
  dragonfly_free(d);


  // float* res = dragonfly(dim, 500, 500, 5.0, w, rosenblock_fitness);
  float fit = rosenblock_fitness(res, dim);

  printf("found fitness=%f\n", fit);
  for (unsigned int i = 0; i < dim; i++) {
    printf("%f\n", res[i]);
  }
  free(res);
    
  MPI_Finalize();
}
