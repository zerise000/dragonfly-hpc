#include "dragonfly-common.h"
#include "stdlib.h"
#include "string.h"
#include "utils.h"
#include <stdbool.h>

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
}

void weights_step(Weights *w) {
  w->s += w->st;
  w->a += w->at;
  w->c += w->ct;
  w->f += w->ft;
  w->e += w->et;
  w->w += w->wt;
}

Dragonfly dragonfly_new(unsigned int dimensions, unsigned int N, unsigned int chunks, unsigned int chunk_id,
                        unsigned int iterations, float space_size,
                        Weights weights,
                        float (*fitness)(float *, unsigned int)) {
  // computes weigths progression
  unsigned int chunk_size=N/chunks;
  if (chunk_id==chunks-1){
    // if it is the last chunk, extend it to include additional elements as needed
    chunk_size=N-chunk_size*(chunks-1);
  }
  weights_compute_steps(&weights, iterations);
  Dragonfly d = {
      .chunks = chunks,
      .chunks_id = chunk_id,
      .N = chunk_size,

      .dim = dimensions,
      
      .iter = iterations,
      .space_size = space_size,
      .w = weights,
      .fitness = fitness,
  };
  return d;
}

void dragonfly_alloc(Dragonfly *d) {
  // allocate, and init random positions
  unsigned int dim = d->dim;
  unsigned int N = d->N;
  unsigned int space_size = d->space_size;
  d->positions = init_array(N * dim, space_size);
  d->speeds = init_array(N * dim, space_size / 20.0);


  d->S = init_array(dim, 0.0);
  d->A = init_array(dim, 0.0);
  d->C = init_array(dim, 0.0);
  d->F = init_array(dim, 0.0);
  d->E = init_array(dim, 0.0);
  d->W = init_array(dim, 0.0);
  d->delta_pos = init_array(dim, 0.0);
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
}

void dragonfly_compute_step(Dragonfly *d, float *average_speed,
                            float *cumulated_pos, float * food, float * enemy) {
  unsigned int dim = d->dim;
  // for each dragonfly
  for (unsigned int j = 0; j < d->N; j++) {
    float *cur_pos = d->positions + dim * j;
    float *cur_speed = d->speeds + dim * j;

    // compute separation: Si = -sumall(X-Xi)
    memcpy(d->S, cur_pos, sizeof(float) * dim);
    scalar_prod_assign(d->S, -(float)d->N, dim);
    sum_assign(d->S, cumulated_pos, dim);
    scalar_prod_assign(d->S, d->w.s, dim);

    // compute alignament: Ai = avarage(Vi)
    memcpy(d->A, average_speed, sizeof(float) * dim);
    scalar_prod_assign(d->A, d->w.a, dim);

    // compute cohesion: Ci = avarage(Xi)-X
    memcpy(d->C, cumulated_pos, sizeof(float) * dim);
    scalar_prod_assign(d->C, 1.0 / (float)d->N, dim);
    sub_assign(d->C, cur_pos, dim);
    scalar_prod_assign(d->C, d->w.c, dim);

    // food attraction: Fi=X_food - X
    memcpy(d->F, food, sizeof(float) * dim);
    sub_assign(d->F, cur_pos, dim);
    scalar_prod_assign(d->F, d->w.f, dim);

    // enemy repulsion: E=X_enemy+X
    memcpy(d->E, enemy, sizeof(float) * dim);
    sum_assign(d->E, cur_pos, dim);
    scalar_prod_assign(d->E, d->w.e, dim);

    // compute speed = sSi + aAi + cCi + fFi + eEi + w
    scalar_prod_assign(cur_speed, d->w.w, dim);
    sum_assign(cur_speed, d->E, dim);
    sum_assign(cur_speed, d->F, dim);
    sum_assign(cur_speed, d->C, dim);
    sum_assign(cur_speed, d->A, dim);
    sum_assign(cur_speed, d->S, dim);
    float speed = length(cur_speed, dim);
    if (speed > d->space_size / 10.0) {
      scalar_prod_assign(cur_speed, d->space_size / 10.0 / speed, dim);
    }
    
    
    // update current pos
    sum_assign(cur_pos, cur_speed, dim);
  }

  // update weights
  weights_step(&d->w);
}


void message_broadcast(Message *my_value, unsigned int i, unsigned int incr,
                       void *data, int dim,
                       void (*raw_sendrecv)(Message *, unsigned int, Message *,
                                            unsigned int, void *)) {
  
  Message recv_buffer;
  int index_other;
  int steps=0;
  int tmp_incr=incr;
  while(tmp_incr>1){
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
  if(my_value->next_food_fitness<recv_buffer.next_food_fitness){
    memcpy(my_value->next_food, recv_buffer.next_food, sizeof(float)*dim);
    my_value->next_food_fitness=recv_buffer.next_food_fitness;
  }
  if(my_value->next_enemy_fitness>recv_buffer.next_enemy_fitness){
    memcpy(my_value->next_enemy, recv_buffer.next_enemy, sizeof(float)*dim);
    my_value->next_enemy_fitness=recv_buffer.next_enemy_fitness;
  }
  my_value->n += recv_buffer.n;
}