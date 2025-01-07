#include "dragonfly-common.h"
#include "utils.h"
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  float cumulated_pos[50];
  float cumulated_speeds[50];
  unsigned int n;
} Message;

void Build_mpi_type_testsend(MPI_Datatype* data){
    int blocklengths[3] ={50, 50, 1};
    MPI_Datatype types[3] ={MPI_FLOAT, MPI_FLOAT, MPI_UNSIGNED};
    MPI_Aint displacements[3];
    displacements[0]=offsetof(Message, cumulated_pos);
    displacements[1]=offsetof(Message, cumulated_speeds);
    displacements[2]=offsetof(Message, n);
    MPI_Type_create_struct(3, blocklengths, displacements, types, data);
    MPI_Type_commit(data);
}

void raw_sendrecv(Message *send, unsigned int destination, Message *recv_buffer,
unsigned int source, void *data_raw){ MPI_Datatype* data_type= data_raw;
    MPI_Sendrecv(
        send,
        1,
        *data_type,
        destination,
        0,
        recv_buffer,
        1,
        *data_type,
        source,
        0,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );
}

void message_broadcast(Message *my_value, unsigned int i, unsigned int incr,
                       void *data, int dim,
                       void (*raw_sendrecv)(Message *, unsigned int, Message *,
                                            unsigned int, void *)) {
  Message recv_buffer;
  int index_other;
  int steps=0;
  int tmp_incr=incr;
  while(tmp_incr>0){
    tmp_incr/=2;
    steps++;
  }
  if ((i>>steps) % 2 == 0) {
    index_other = i + incr;
  } else {
    index_other = i - incr;
  }

  raw_sendrecv(my_value, index_other, &recv_buffer, index_other, data);
  sum_assign(my_value->cumulated_pos, recv_buffer.cumulated_pos, dim);
  sum_assign(my_value->cumulated_speeds, recv_buffer.cumulated_speeds, dim);
  my_value->n += recv_buffer.n;
}

// take timing not including IO
float *dragonfly_compute(Dragonfly *d, unsigned int chunks, unsigned int dim, 
                         unsigned int iter) {
  unsigned int joint_chunks = 1;
  
  Message *messages = malloc(sizeof(Message) * chunks);
  // for each iteration
  for (unsigned int i = 0; i < iter; i++) {
    
    // compute avarage speed and positions
    for (unsigned int j = 0; j < chunks; j++) {
      zeroed(messages[j].cumulated_pos, dim);
      zeroed(messages[j].cumulated_speeds, dim);
      messages[j].n=0;
      for (unsigned int k = 0; k < d[j].N; k++) {
        sum_assign(messages[j].cumulated_pos, d[j].positions + dim * k, dim);
        sum_assign(messages[j].cumulated_speeds, d[j].speeds + dim * k, dim);
        //sum_assign(messages[j].cumulated_speeds, d[0].speeds + dim * j, dim);
        messages[j].n += 1;
      }
    }
    // computed, then broadcast to others

    // execute log2(joint_chunks)
    for (unsigned int s = 1; s < joint_chunks; s *= 2) {
      for (unsigned int j = 0; j < chunks; j++) {
        message_broadcast(&messages[j], j, s, messages, dim, raw_sendrecv);
      }
    }

    //compute average
    for (unsigned int j = 0; j < chunks; j++) {
      scalar_prod_assign(messages[j].cumulated_speeds, 1.0 / (float)messages[j].n, dim);
      printf("test %d\n", messages[j].n);//
    }
    // compute one step
    dragonfly_compute_step(d, d->average_speed, d->cumulated_pos);
  }
  free(messages);
  return d->next_food;
}

int main() {
  //srand(time(NULL));
  srand(1);
  Weights w = {
      // exploring
      .al = {0.3, 0.01},
      .cl = {0.01, 0.3},
      // swarming
      .sl = {0.1, 0.1},
      .fl = {0.1, 0.1},
      .el = {0.1, 0.1},
      .wl = {0.1, 0.1},
  };
  unsigned int dim = 2;
  unsigned int chunks = 4;
  unsigned int iterations=500;
  unsigned int N=500;
  Dragonfly *d = malloc(sizeof(Dragonfly) * chunks);
  for (unsigned int i = 0; i < chunks; i++) {
    d[i] = dragonfly_new(dim, N, chunks, i, iterations, 5.0, w, rosenblock_fitness);
    dragonfly_alloc(&d[i]);
  }

  float *tmp = dragonfly_compute(d, chunks, dim, iterations);
  float res[20];
  memcpy(res, tmp, sizeof(float) * dim);
  for(unsigned int i =0; i<chunks; i++){
    dragonfly_free(d[i]);
  }
  free(d);
  
  // float* res = dragonfly(dim, 500, 500, 5.0, w, rosenblock_fitness);
  float fit = sphere_fitness(res, dim);

  printf("found fitness=%f\n", fit);
  for (unsigned int i = 0; i < dim; i++) {
    printf("%f\n", res[i]);
  }
}
