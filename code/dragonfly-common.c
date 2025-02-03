#include "dragonfly-common.h"
#include "stdlib.h"
#include "string.h"
#include "utils.h"
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>

// n, chunks, iterations, dim
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
  if(p.chunks> p.n){
    fprintf(stderr, "chunk must be smaller than n");
    exit(-1);
  }
  return p;
};

void weights_compute_steps(Weights *w, unsigned int steps) {
  w->st = (w->sl[1] - w->sl[0]) / (float)steps;
  w->s = w->sl[0];
  w->at = (w->al[1] - w->al[0]) / (float)steps;
  w->a = w->al[0];

  w->ct = (w->cl[1] - w->cl[0]) / (float)steps;
  w->c = w->cl[0];
  w->ft = (w->fl[1] - w->fl[0]) / (float)steps;
  w->f = w->fl[0];

  w->et = (w->el[1] - w->el[0]) / (float)steps;
  w->e = w->el[0];

  w->wt = (w->wl[1] - w->wl[0]) / (float)steps;
  w->w = w->wl[0];

  w->lt = (w->ll[1] - w->ll[0]) / (float)steps;
  w->l = w->ll[0];
}

void weights_step(Weights *w) {
  w->s += w->st;
  w->a += w->at;
  w->c += w->ct;
  w->f += w->ft;
  w->e += w->et;
  w->w += w->wt;
  w->l += w->lt;
}

Dragonfly dragonfly_new(unsigned int dimensions, unsigned int N,
                        unsigned int iterations, float space_size,
                        Weights weights,
                        float (*fitness)(float *, unsigned int), unsigned int random_seed) {
  // computes weigths progression

  weights_compute_steps(&weights, iterations);

  Dragonfly d = {
      .N = N,
      .dim = dimensions,
      
      .space_size = space_size,
      .w = weights,
      .fitness = fitness,
      .seed = random_seed,
  };
  return d;
}

void dragonfly_alloc(Dragonfly *d) {
  // allocate, and init random positions
  unsigned int dim = d->dim;
  unsigned int N = d->N;
  unsigned int space_size = d->space_size;
  d->positions = init_array(N * dim, space_size, &d->seed);
  d->speeds = init_array(N * dim, space_size / 20.0, &d->seed);


  d->S = init_array(dim, 0.0, &d->seed);
  d->A = init_array(dim, 0.0, &d->seed);
  d->C = init_array(dim, 0.0, &d->seed);
  d->F = init_array(dim, 0.0, &d->seed);
  d->E = init_array(dim, 0.0, &d->seed);
  d->W = init_array(dim, 0.0, &d->seed);
  d->levy = init_array(dim, 0.0, &d->seed);
  d->delta_pos = init_array(dim, 0.0, &d->seed);
}

void dragonfly_free(Dragonfly d) {
  free(d.positions);
  free(d.speeds);
  free(d.S);
  free(d.A);
  free(d.C);
  free(d.F);
  free(d.E);
  free(d.W);
  free(d.delta_pos);
  free(d.levy);
}



void raw_send_recv(Message *send, unsigned int destination, Message *recv_buffer,
                  unsigned int source, MPI_Datatype *data_type) {
  MPI_Sendrecv(send, 1, *data_type, destination, 0, recv_buffer, 1, *data_type,
               source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}


void message_broadcast(Message *message, unsigned int index, int n, MPI_Datatype *data_type){
  
  Message recv_buffer;
  for(unsigned int steps=0; (1<<steps)<n; steps++){
    unsigned int index_other;
    if ((index>>steps) % 2 == 0) {
      index_other = index+(1<<steps);
    } else {
      index_other = index-(1<<steps);
    }
    raw_send_recv(message, index_other, &recv_buffer, index_other, data_type);
    //TODO check
    memcpy(&message->status[recv_buffer.start_chunk], &recv_buffer.status[recv_buffer.start_chunk], sizeof(ComputationStatus)*(recv_buffer.end_chunk-recv_buffer.start_chunk));
    message->start_chunk = min(message->start_chunk, recv_buffer.start_chunk);
    message->end_chunk = max(message->end_chunk, recv_buffer.end_chunk);
  }
}

void computation_status_merge(ComputationStatus *out, ComputationStatus *in, unsigned int dim){
  sum_assign(out->cumulated_pos, in->cumulated_pos, dim);
  sum_assign(out->cumulated_speeds, in->cumulated_speeds, dim);
  if(out->next_food_fitness<in->next_food_fitness){
    memcpy(out->next_food, in->next_food, sizeof(float)*dim);
    out->next_food_fitness=in->next_food_fitness;
  }
  if(out->next_enemy_fitness>in->next_enemy_fitness){
    memcpy(out->next_enemy, in->next_enemy, sizeof(float)*dim);
    out->next_enemy_fitness=in->next_enemy_fitness;
  }
  out->n += in->n;
}