#include "dragonfly-common.h"
#include "stdlib.h"
#include "string.h"
#include "utils.h"
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
  if((p.chunks&(p.chunks-1))!=0){
    fprintf(stderr, "chunk must be a power of two");
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

  w->max_speedt = (w->max_speedl[1] - w->max_speedl[1]) / (float)steps;
  w->max_speed = w->max_speedl[0];
}

void weights_step(Weights *w) {
  w->s += w->st;
  w->a += w->at;
  w->c += w->ct;
  w->f += w->ft;
  w->e += w->et;
  w->w += w->wt;
  w->l += w->lt;
  w->max_speed += w->max_speedt;
}

Dragonfly dragonfly_new(unsigned int dimensions, unsigned int N, unsigned int chunks, unsigned int chunk_id,
                        unsigned int iterations, float space_size,
                        Weights weights,
                        float (*fitness)(float *, unsigned int), unsigned int random_seed) {
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

void dragonfly_compute_step(Dragonfly *d, float *average_speed,
                            float *cumulated_pos, float * food, float * enemy, unsigned int N) {
  unsigned int dim = d->dim;
  // for each dragonfly
  for (unsigned int j = 0; j < d->N; j++) {
    float *cur_pos = d->positions + dim * j;
    float *cur_speed = d->speeds + dim * j;

    // compute separation: Si = -sumall(X-Xi)
    memcpy(d->S, cur_pos, sizeof(float) * dim);
    scalar_prod_assign(d->S, -(float)N, dim);
    sum_assign(d->S, cumulated_pos, dim);
    scalar_prod_assign(d->S, d->w.s, dim);

    // compute alignament: Ai = avarage(Vi)
    memcpy(d->A, average_speed, sizeof(float) * dim);
    scalar_prod_assign(d->A, d->w.a, dim);

    // compute cohesion: Ci = avarage(Xi)-X
    memcpy(d->C, cumulated_pos, sizeof(float) * dim);
    scalar_prod_assign(d->C, 1.0 / (float)N, dim);
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

    brownian_motion(d->levy, dim, &d->seed);
    

    // compute speed = sSi + aAi + cCi + fFi + eEi + w
    scalar_prod_assign(cur_speed, d->w.w, dim);
    sum_assign(cur_speed, d->E, dim);
    sum_assign(cur_speed, d->F, dim);
    sum_assign(cur_speed, d->C, dim);
    sum_assign(cur_speed, d->A, dim);
    sum_assign(cur_speed, d->S, dim);
    sum_assign(cur_speed, d->levy, dim);
    float speed = length(cur_speed, dim);
    if (speed > d->w.max_speed) {
      scalar_prod_assign(cur_speed, d->w.max_speed / speed , dim);
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

void message_acumulate(Message *message, Dragonfly *d, float* best, float* best_fitness){
  unsigned int dim=d->dim;
  zeroed(message->cumulated_pos, dim);
  zeroed(message->cumulated_speeds, dim);
  memcpy(message->next_enemy, d->positions, sizeof(float) * dim);
  memcpy(message->next_food, d->positions, sizeof(float) * dim);
  message->next_enemy_fitness =
      d->fitness(message->next_enemy, dim);
  message->next_food_fitness = d->fitness(message->next_food, dim);
  message->n = 0;
  for (unsigned int k = 0; k < d->N; k++) {
    float *cur_pos = d->positions + dim * k;
    sum_assign(message->cumulated_pos, cur_pos, dim);
    sum_assign(message->cumulated_speeds, d->speeds + dim * k, dim);
    float fitness = d->fitness(cur_pos, dim);
    if (fitness > message->next_food_fitness) {
      memcpy(message->next_food, cur_pos, sizeof(float) * dim);
      message->next_food_fitness = fitness;
    }
    if (fitness < message->next_enemy_fitness) {
      memcpy(message->next_enemy, cur_pos, sizeof(float) * dim);
      message->next_enemy_fitness = fitness;
    }
    if (fitness > *best_fitness) {
      memcpy(best, cur_pos, sizeof(float) * dim);
      *best_fitness = fitness;
    }
    message->n += 1;
  }
}