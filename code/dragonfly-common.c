#include "dragonfly-common.h"
#include "stdlib.h"
#include "string.h"
#include "utils-special.h"
#include "utils.h"

#include <stdbool.h>
#include <stdio.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

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
                        Fitness fitness, unsigned int random_seed) {
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
#ifdef USE_MPI

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
    memcpy(&message->status[recv_buffer.start_chunk], &recv_buffer.status[recv_buffer.start_chunk], sizeof(ComputationStatus)*(recv_buffer.end_chunk-recv_buffer.start_chunk));
    message->start_chunk = min(message->start_chunk, recv_buffer.start_chunk);
    message->end_chunk = max(message->end_chunk, recv_buffer.end_chunk);
  }
}




void Build_mpi_type_computation_status(MPI_Datatype *data) {
  int blocklengths[7] = {MESSAGE_SIZE, MESSAGE_SIZE, MESSAGE_SIZE, MESSAGE_SIZE, 1, 1, 1};
  MPI_Datatype types[7] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT,   MPI_FLOAT,
                           MPI_FLOAT, MPI_FLOAT, MPI_UNSIGNED};
  MPI_Aint displacements[7];
  displacements[0] = offsetof(ComputationStatus, cumulated_pos);
  displacements[1] = offsetof(ComputationStatus, cumulated_speeds);
  displacements[2] = offsetof(ComputationStatus, next_enemy);
  displacements[3] = offsetof(ComputationStatus, next_food);
  displacements[4] = offsetof(ComputationStatus, next_enemy_fitness);
  displacements[5] = offsetof(ComputationStatus, next_food_fitness);
  displacements[6] = offsetof(ComputationStatus, n);
  MPI_Type_create_struct(7, blocklengths, displacements, types, data);
  MPI_Type_commit(data);
}



void build_mpi_type_message(MPI_Datatype *out, MPI_Datatype *computation_status){

  //typedef struct{
  //  unsigned int start_chunk, end_chunk;
  //  ComputationStatus status[MAX_CHUNKS];
  //} Message;

  int blocklengths[3] = {1, 1, MAX_CHUNKS};
  MPI_Datatype types[3] = {MPI_UNSIGNED, MPI_UNSIGNED, *computation_status};
  MPI_Aint displacements[3];
  displacements[0] = offsetof(Message, start_chunk);
  displacements[1] = offsetof(Message, end_chunk);
  displacements[2] = offsetof(Message, status);
  MPI_Type_create_struct(3, blocklengths, displacements, types, out);
  MPI_Type_commit(out);
}
#endif




// take timing not including IO
float *dragonfly_compute(Parameters p, Weights w, Fitness fitness, unsigned int threads, unsigned int rank_id, float space_size, unsigned int srand) {
  if(MESSAGE_SIZE<p.dim){
    fprintf(stderr, "impossible to compute with %d dimensions, recompile with a bigger MESSAGE_SIZE\n", p.dim);
    exit(-1);
  }
  if(p.chunks>MAX_CHUNKS){
    fprintf(stderr, "impossible to compute with %d chunks, recompile with a bigger MAX_CHUNKS\n", p.chunks);
    exit(-1);
  }
  if(p.chunks==0){
    fprintf(stderr, "impossible to compute with 0 chunks\n");
    exit(-1);
  }

  #ifndef USE_MPI
    if(threads>1){
      fprintf(stderr, "threads>1, recompile with mpi");
      exit(-1);
    }
  #endif
  // compute start and end rank
  unsigned int start_chunk = rank_id*(p.chunks/threads) + min(rank_id, p.chunks%threads);
  unsigned int current_chunks = p.chunks/threads + (rank_id<(p.chunks%threads));
  unsigned int end_chunk = start_chunk + current_chunks;
  

  #ifdef USE_MPI
    MPI_Datatype message_type, computation_status_type;
    Build_mpi_type_computation_status(&computation_status_type);
    build_mpi_type_message(&message_type, &computation_status_type);
  #endif

  Dragonfly *d = malloc(sizeof(Dragonfly) * (current_chunks));

  // allocate problem
  for (unsigned int i = 0; i < current_chunks; i++) {
    unsigned int cur_n = p.n/p.chunks + (i<(p.n%p.chunks));
    d[i] = dragonfly_new(p.dim, cur_n, p.iterations, space_size, w,
                         fitness, i + start_chunk+ srand);
    dragonfly_alloc(&d[i]);
  }


  float *best = malloc(sizeof(float) * p.dim);
  memcpy(best, d->positions, p.dim * sizeof(float));
  float best_fitness = d->fitness(best, &d->seed, p.dim);

  unsigned int joint_chunks = 1;
  int log_chunks = 0;
  int tmp_chunks = p.chunks;
  while (tmp_chunks > 1) {
    tmp_chunks /= 2;
    log_chunks++;
  }

  unsigned int update_chunk_steps = (p.iterations + log_chunks) / (log_chunks + 1);
  Message message;

  // for each iteration
  for (unsigned int i = 0; i < p.iterations; i++) {
    if (i != 0 && i % update_chunk_steps == 0) {
      joint_chunks *= 2;
    }

    // compute avarage speed and positions
    for(unsigned int j=0; j<current_chunks; j++){
      computation_accumulate(&message.status[j+start_chunk], &d[j], best, &best_fitness);
    }
    message.start_chunk=start_chunk;
    message.end_chunk=end_chunk;
    
    //send current chunks to others, and retrive collective status (if USE_MPI is defined)
    #ifdef USE_MPI
      message_broadcast(&message, rank_id, threads, &message_type);
    #endif

    //prepare and compute
    for(unsigned int j=start_chunk-start_chunk%joint_chunks; j<end_chunk; j+=joint_chunks){
      for(unsigned int k=1; k<joint_chunks; k++){
        if(j+k<p.chunks){
          computation_status_merge(&message.status[j], &message.status[j+k], p.dim);
        }
      }
      scalar_prod_assign(message.status[j].cumulated_speeds,
                         1.0 / (float)message.status[j].n, p.dim);
      for(unsigned int k=0; k<joint_chunks; k++){
        if(start_chunk<=j+k && j+k<end_chunk){
          dragonfly_compute_step(&d[j+k-start_chunk], message.status[j].cumulated_speeds,
                            message.status[j].cumulated_pos, message.status[j].next_food,
                            message.status[j].next_enemy, message.status[j].n);
        }
      }
    }
  }
  // check last iteration
  // compute avarage speed and positions
  for(unsigned int j=0; j<current_chunks; j++){
      computation_accumulate(&message.status[j+start_chunk], &d[j], best, &best_fitness);
    }
  message.start_chunk=start_chunk;
  message.end_chunk=end_chunk;

  if(message.status[start_chunk].next_food_fitness< best_fitness) {
    memcpy(message.status[start_chunk].next_food, best, p.dim*sizeof(float));
    message.status[start_chunk].next_food_fitness = best_fitness;
  }

  //send current chunks to others, and retrive collective status (if USE_MPI is defined)
  #ifdef USE_MPI
    message_broadcast(&message, rank_id, threads, &message_type);
  #endif

  for(unsigned int i=0; i<p.chunks; i++){
    computation_status_merge(&message.status[0], &message.status[i], p.dim);
  }

  memcpy(best, message.status[start_chunk].next_food, p.dim*sizeof(float));

  //memcpy(best, message.status[0].next_food, p.dim*sizeof(float));
  #ifdef USE_MPI
  MPI_Type_free(&message_type);
  #endif
  return best;
}