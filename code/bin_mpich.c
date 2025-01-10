#include "dragonfly-common.h"
#include "utils.h"
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

float *dragonfly_compute(Dragonfly *d, unsigned int chunks, unsigned int dim,
                         unsigned int iter);

int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  // wait for all the process to start
  MPI_Barrier(MPI_COMM_WORLD);

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
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Fitness fitness = rastrigin_fitness;
  if ((int) p.chunks!=comm_size){
    fprintf(stderr, "chunks!=comm_size (%d!=%d)", p.chunks, comm_size);
  }
  
  
  srand(rank);
  Dragonfly d = dragonfly_new(p.dim, p.n, p.chunks,  rank, p.iterations, 5.0, w,
                         fitness);
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


// take timing not including IO
float *dragonfly_compute(Dragonfly *d, unsigned int chunks, unsigned int dim,
                         unsigned int iter) {

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
    message_acumulate(&message, d, best, &best_fitness);
    // computed, then broadcast to others

    // execute log2(joint_chunks)
    for (unsigned int s = 1; s < joint_chunks; s *= 2) {
      message_broadcast(&message, d->chunks_id, s, &message_type, dim, raw_sendrecv);
    }

    // prepare and compute step
    scalar_prod_assign(message.cumulated_speeds,
                         1.0 / (float)message.n, dim);
    dragonfly_compute_step(d, message.cumulated_speeds,
                            message.cumulated_pos, message.next_food,
                            message.next_enemy, d->N);
  }
  // check last iteration

  message_acumulate(&message, d, best, &best_fitness);
  if(best_fitness>message.next_food_fitness){
    memcpy(message.next_food, best, dim*sizeof(float));
    message.next_food_fitness=best_fitness;
  }
  for (unsigned int s = 1; s < chunks; s *= 2) {
    message_broadcast(&message, d->chunks_id, s, &message_type, dim, raw_sendrecv);
  }
  memcpy(best, message.next_food, dim*sizeof(float));
  
  MPI_Type_free(&message_type);
  return best;
}

