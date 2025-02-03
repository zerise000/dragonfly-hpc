#include <stddef.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>
#include <threads.h>
#include <unistd.h>
#include <time.h>

#include "dragonfly-common.h"
#include "utils.h"
#include "utils-special.h"

#define USE_MPI

float *dragonfly_compute(Parameters p, Weights w, Fitness fitness, unsigned int threads, unsigned int rank_id, unsigned int srand);

#ifndef DA_MPICH_LIB
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
      .al = {0.1, 0.1},
      .cl = {0.7, 0.7},
      // swarming
      .sl = {0.1, 0.1},
      .fl = {1.0, 1.0},
      .el = {0.0, 0.0},
      .wl = {0.6, 0.6},
      .ll = {0.1, 0.1},
  };
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Fitness fitness = rastrigin_fitness;
  float * res = dragonfly_compute(p, w, fitness, comm_size, rank, 0);
  
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
#endif

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





// take timing not including IO
float *dragonfly_compute(Parameters p, Weights w, Fitness fitness, unsigned int threads, unsigned int rank_id, unsigned int srand) {
  if(MESSAGE_SIZE<p.dim){
    fprintf(stderr, "impossible to compute with %d dimensions, recompile with a bigger MESSAGE_SIZE\n", p.dim);
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
  if(p.chunks>MAX_CHUNKS){
    fprintf(stderr, "impossible to compute with %d chunks, recompile with a bigger MAX_CHUNKS\n", p.chunks);
    exit(-1);
  }

  #ifdef USE_MPI
    MPI_Datatype message_type, computation_status_type;
    Build_mpi_type_computation_status(&computation_status_type);
    build_mpi_type_message(&message_type, &computation_status_type);
  #endif

  Dragonfly *d = malloc(sizeof(Dragonfly) * (current_chunks));

  // allocate problem
  for (unsigned int i = 0; i < current_chunks; i++) {
    unsigned int cur_n = p.n/p.chunks + (i<(p.n%p.chunks));
    d[i] = dragonfly_new(p.dim, cur_n, p.iterations, 100.0, w,
                         fitness, i + start_chunk+ srand);
    dragonfly_alloc(&d[i]);
  }


  float *best = malloc(sizeof(float) * p.dim);
  memcpy(best, d->positions, p.dim * sizeof(float));
  float best_fitness = d->fitness(best, p.dim);

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
  
  MPI_Type_free(&message_type);
  return best;
}

