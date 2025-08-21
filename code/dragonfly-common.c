#include "dragonfly-common.h"
#include "stdlib.h"
#include "string.h"
#include "utils.h"

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
// #ifdef USE_MPI
#include <mpi.h>
#include <unistd.h>
// #endif

// n, chunks, iterations, dim
Parameters parameter_parse(int argc, char *argv[]) {
  Parameters p;
  if (argc != 6) {
    fprintf(stderr, "invalid parameter count: expected population_size, "
                    "starting_chunk_count, iterations, problem_dimensions, "
                    "threads_per_process");
    exit(-1);
  }
  p.population_size = atoi(argv[1]);
  p.starting_chunk_count = atoi(argv[2]);
  p.iterations = atoi(argv[3]);
  p.problem_dimensions = atoi(argv[4]);

  p.threads_per_process = atoi(argv[5]);

  if (p.population_size == 0 || p.starting_chunk_count == 0 ||
      p.iterations == 0 || p.problem_dimensions == 0 ||
      p.threads_per_process == 0) {
    fprintf(stderr,
            "invalid parameter they must be all bigger than 1 (and integers)");
    exit(-1);
  }
  if (p.starting_chunk_count > p.population_size) {
    fprintf(stderr, "chunk must be smaller than n");
    exit(-1);
  }
  return p;
};

// function used to compute the step for the linear progression of the weights.
void weights_compute_steps(Weights *w, unsigned int steps) {
  // compute step increase
  w->st = (w->sl[1] - w->sl[0]) / (float)steps;
  // set initial value
  w->s = w->sl[0];
  // repeat for the othe values
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

// compute one step forward for the weights
void weights_step(Weights *w) {
  w->s += w->st;
  w->a += w->at;
  w->c += w->ct;
  w->f += w->ft;
  w->e += w->et;
  w->w += w->wt;
  w->l += w->lt;
}

// initialize a new dragonfly instance, it does not alloc the memory.
Dragonfly dragonfly_new(unsigned int dimensions, unsigned int start,
                        unsigned int end, unsigned int iterations,
                        float space_size, Weights weights, Fitness fitness,
                        unsigned int random_seed) {
  // computes weigths progression

  weights_compute_steps(&weights, iterations);

  Dragonfly d = {

      .start = start,
      .end = end,
      .dim = dimensions,
      .space_size = space_size,
      .w = weights,
      .fitness = fitness,
      .seed = random_seed,
  };
  return d;
}

// allocates the buffers for the dragonfly algorithm
void dragonfly_alloc(Dragonfly *d) {
  // allocate, and init random positions
  unsigned int dim = d->dim;
  unsigned int N = d->end - d->start;
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

// free al buffer for the dragonfly algorithm
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

void computation_status_merge(ComputationStatus *out, ComputationStatus *in,
                              unsigned int dim) {
  sum_assign(out->cumulated_pos, in->cumulated_pos, dim);
  sum_assign(out->cumulated_speeds, in->cumulated_speeds, dim);
  if (out->next_food_fitness < in->next_food_fitness) {
    memcpy(out->next_food, in->next_food, sizeof(float) * dim);
    out->next_food_fitness = in->next_food_fitness;
  }
  if (out->next_enemy_fitness > in->next_enemy_fitness) {
    memcpy(out->next_enemy, in->next_enemy, sizeof(float) * dim);
    out->next_enemy_fitness = in->next_enemy_fitness;
  }
  out->n += in->n;
}

// #ifdef USE_MPI
/*
void raw_send_recv(Message *send, unsigned int destination, Message
*recv_buffer, unsigned int source, MPI_Datatype *data_type) { MPI_Sendrecv(send,
1, *data_type, destination, 0, recv_buffer, 1, *data_type, source, 0,
MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}


void message_broadcast(Message *message, unsigned int index, int n, MPI_Datatype
*data_type){

  Message recv_buffer;
  for(unsigned int steps=0; (1<<steps)<n; steps++){
    unsigned int index_other;
    if ((index>>steps) % 2 == 0) {
      index_other = index+(1<<steps);
    } else {
      index_other = index-(1<<steps);
    }
    raw_send_recv(message, index_other, &recv_buffer, index_other, data_type);
    memcpy(&message->status[recv_buffer.start_chunk],
&recv_buffer.status[recv_buffer.start_chunk],
sizeof(ComputationStatus)*(recv_buffer.end_chunk-recv_buffer.start_chunk));
    message->start_chunk = min(message->start_chunk, recv_buffer.start_chunk);
    message->end_chunk = max(message->end_chunk, recv_buffer.end_chunk);
  }
}*/

void Build_mpi_type_computation_status(MPI_Datatype *data) {
  int blocklengths[7] = {
      MESSAGE_SIZE, MESSAGE_SIZE, MESSAGE_SIZE, MESSAGE_SIZE, 1, 1, 1};
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

// #endif

typedef struct {
  unsigned int start;
  unsigned int end;
  bool to_shift_left;
  bool to_shift_right;
  bool complete;

  ComputationStatus comp;
} LogicalChunk;

unsigned int assing_logical_chunks(unsigned int phisical_chunk_start,
                                   unsigned int phisical_chunk_end,
                                   unsigned int phisical_chunk_size,
                                   unsigned int population_size,
                                   unsigned int remaining,
                                   unsigned int logical_chunk_size,
                                   LogicalChunk *buffer) {
  unsigned int cur_start =
      (phisical_chunk_start / logical_chunk_size) * logical_chunk_size;
  unsigned int index = 0;
  while (cur_start < phisical_chunk_end) {
    unsigned int cur_end = min(cur_start + logical_chunk_size, population_size);
    bool starts_in_previous = cur_start < phisical_chunk_start &&
                              phisical_chunk_start < cur_end &&
                              cur_end < phisical_chunk_end;
    unsigned int next_phisical_end =
        remaining <= 1 ? population_size
                       : phisical_chunk_end + phisical_chunk_size;
    bool ends_in_next =
        phisical_chunk_end < cur_end && cur_end < next_phisical_end;
    LogicalChunk cur = {
        .start = cur_start,
        .end = cur_end,
        .to_shift_left = starts_in_previous && !ends_in_next,
        .to_shift_right = !starts_in_previous && ends_in_next,
        .complete = !starts_in_previous && !ends_in_next,
    };
    // printf("chunk: %d-%d %d-%d %d %d %d\n", phisical_chunk_start,
    //          phisical_chunk_end, cur_start, cur_end, cur.to_shift_left,
    //          cur.to_shift_right, index);
    buffer[index] = cur;
    index++;
    cur_start += logical_chunk_size;
  }
  return index;
}
unsigned int upper_log2(unsigned int n) {
  unsigned int log = 0;
  if (n == 0)
    return 0; // or handle as needed
  n--;
  while (n > 0) {
    n >>= 1;
    log++;
  }
  return log;
}
void comunicate(LogicalChunk *cur, MPI_Datatype type,
                unsigned int phisical_chunk_size, unsigned int rank_id,
                unsigned int thread_count, unsigned int population_size,
                unsigned int dim) {
  unsigned int start = cur->start / phisical_chunk_size;
  unsigned int end = min(cur->end / phisical_chunk_size, thread_count);
  if (end == thread_count && cur->end < population_size) {
    end -= 1;
  }
  unsigned int size = end - start;

  unsigned int required_steps = upper_log2(size);

  ComputationStatus tmp;
  unsigned inner_rank_id = rank_id - start;
  // printf("comunicate1 %d %d %d %d %d\n", cur->start, cur->end, size, rank_id,
  // required_steps);
  for (unsigned int i = 0; i < required_steps; i++) {

    unsigned partner = (inner_rank_id ^ (1 << i)) + start;

    if (start <= partner && partner < end) {
      // printf("comunicate2 %d %d %d->%d\n", start, end, rank_id, partner);
      // fflush(stdout);
      MPI_Sendrecv(&cur->comp, 1, type, partner, 1, &tmp, 1, type, partner, 1,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      computation_status_merge(&cur->comp, &tmp, dim);
    }
  }

  if (required_steps <= 1) {
    return;
  }

  for (int i = required_steps - 2; i >= 0; i--) {
    unsigned partner = (inner_rank_id ^ (1 << i)) + start;
    if (start <= partner && partner < end) {
      // printf("comunicate3 %d %d %d->%d\n", start, end, rank_id, partner);
      // fflush(stdout);
      MPI_Sendrecv(&cur->comp, 1, type, partner, 1, &tmp, 1, type, partner, 1,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // computation_status_merge(&cur->comp, &tmp, dim);
      if (cur->comp.n < tmp.n) {
        memcpy(&cur->comp, &tmp, sizeof(ComputationStatus));
      }
    }
  }

  // printf("end comunicate1 %d %d %d %d %d\n", cur->start, cur->end, size,
  // rank_id, required_steps);
}

// TODO paralelize
// TODO move to right place
//  it computes the best, the food, the enemy, and the sums of speeds and
//  positions of an interval. the interval must be inside the current thread
//  chunk
void new_computation_accumulate(Dragonfly *d, LogicalChunk *current_chunk,
                                unsigned int *seed) {

  unsigned start = max(current_chunk->start, d->start);
  unsigned end = min(current_chunk->end, d->end);

  unsigned dim = d->dim;
  float *cumulated_pos = current_chunk->comp.cumulated_pos;
  float *cumulated_speed = current_chunk->comp.cumulated_speeds;

  zeroed(cumulated_pos, dim);
  zeroed(cumulated_speed, dim);

  float next_enemy_fitness = d->fitness(d->positions, &d->seed, dim);
  float next_food_fitness = next_enemy_fitness;
  // status->n = d->local_n;

  unsigned int indexes[2] = {0, 0};
  end = end - d->start;
  for (unsigned int k = start - d->start; k < end; k++) {
    float *iter_pos = d->positions + dim * k;
    float *iter_speed = d->speeds + dim * k;
    sum_assign(cumulated_pos, iter_pos, dim);
    sum_assign(cumulated_speed, iter_speed, dim);
    // printf("SUM_ASSIGN %f\n", *cumulated_pos);

    float fitness = d->fitness(iter_pos, seed, dim);

    if (fitness > next_food_fitness) {
      indexes[0] = k;
      next_food_fitness = fitness;
    }
    if (fitness < next_enemy_fitness) {
      indexes[1] = k;
      next_enemy_fitness = fitness;
    }
  }

  memcpy(current_chunk->comp.next_food, d->positions + indexes[0] * dim,
         sizeof(float) * dim);
  memcpy(current_chunk->comp.next_enemy, d->positions + indexes[1] * dim,
         sizeof(float) * dim);
  current_chunk->comp.next_enemy_fitness = next_enemy_fitness;
  current_chunk->comp.next_food_fitness = next_food_fitness;
  current_chunk->comp.n = end - (start - d->start);
}

void inner_dragonfly_step(Dragonfly *d, LogicalChunk chunk,
                          unsigned int random_seed) {

  unsigned int dimensions = d->dim;
  printf("%d\n", dimensions);
  float S;
  float A;
  float C;
  float F;
  float E;
  float levy;

  float *cur_pos;
  float *cur_speed;

  unsigned start = max(d->start, chunk.start) - d->start;
  unsigned end = min(d->end, chunk.end) - d->start;

  unsigned N = chunk.comp.n;

  for (unsigned int j = start; j < end; j++) {
    unsigned random = random_seed + j;
    cur_pos = d->positions + j * dimensions;
    cur_speed = d->speeds + j * dimensions;

    // compute speed = sSi + aAi + cCi + fFi + eEi + w

    for (unsigned int i = 0; i < dimensions; i++) {
      S = (chunk.comp.cumulated_pos[i] / ((float)N)) - cur_pos[i];
      A = chunk.comp.cumulated_speeds[i];
      C = (chunk.comp.cumulated_pos[i] / (float)N) - cur_pos[i];
      F = chunk.comp.next_food[i] - cur_pos[i];
      E = chunk.comp.next_enemy[i] + cur_pos[i];
      levy = RAND_FLOAT(1.0, &random);

      cur_speed[i] *= d->w.w;
      cur_speed[i] += d->w.s * S;
      cur_speed[i] += d->w.a * A;
      cur_speed[i] += d->w.c * C;
      cur_speed[i] += d->w.f * F;
      cur_speed[i] += d->w.e * E;
      cur_speed[i] += levy;

      cur_pos[i] += cur_speed[i];
    }
  }
}

float *dragonfly_compute(Parameters p, Weights w, ChunkSize c, Fitness fitness,
                         unsigned int threads, unsigned int rank_id,
                         float space_size, unsigned int srand) {
  if (MESSAGE_SIZE < p.problem_dimensions) {
    fprintf(stderr,
            "impossible to compute with %d dimensions, recompile with a bigger "
            "MESSAGE_SIZE\n",
            p.problem_dimensions);
    exit(-1);
  }
#ifndef USE_MPI
  if (threads > 1) {
    fprintf(stderr, "impossible to compute with multiple process without mpi, "
                    "recompile with USE_MPI\n");
    exit(-1);
  }
#endif
  // TODO remove temporary asserts
  assert(p.population_size >= threads);
  assert(p.population_size >= c.start_count);
  // compute start and end phisical_thread
  unsigned int phisical_chunk_size = p.population_size / threads;
  unsigned int phisical_start = phisical_chunk_size * rank_id;
  unsigned int phisical_end = rank_id == threads - 1
                                  ? p.population_size
                                  : phisical_start + phisical_chunk_size;
  // printf("start=%d end=%d rank=%d\n", phisical_start, phisical_end, rank_id);
  assert(phisical_chunk_size >= 1);

  // compute maximum number of logical_chunks in a phisical_chunk, it will not
  // divide in exactly c.start_count, but it tries to be as close and balanced
  // as possible. This means that for high start_count the number could differ
  // (if close to population size)
  unsigned int logical_chunk_size =
      (p.population_size + c.start_count - 1) / c.start_count;

  assert(phisical_chunk_size >= 1);
  assert(logical_chunk_size >= 1);

  // Calculate maximum chunks more safely
  // The assing_logical_chunks function can start before phisical_chunk_start
  // and can create more chunks than expected, so we need a more generous upper
  // bound
  unsigned int logical_start =
      (phisical_start / logical_chunk_size) * logical_chunk_size;
  unsigned int logical_span = phisical_end - logical_start;
  unsigned int maximum_logical_chunks =
      (logical_span + logical_chunk_size - 1) / logical_chunk_size + 1;

  // define mpi structs (if available)
  // #ifdef USE_MPI
  MPI_Datatype computation_status_type;
  Build_mpi_type_computation_status(&computation_status_type);
  // #endif

  LogicalChunk *local_chunks =
      malloc(sizeof(LogicalChunk) * maximum_logical_chunks);

  // Initialize the allocated memory to avoid uninitialized data
  memset(local_chunks, 0, sizeof(LogicalChunk) * maximum_logical_chunks);

  Dragonfly cur =
      dragonfly_new(p.problem_dimensions, phisical_start, phisical_end,
                    p.iterations, space_size, w, fitness, rank_id + srand);

  dragonfly_alloc(&cur);
  ComputationStatus temp_comp;

  float *best = malloc(sizeof(float) * p.problem_dimensions);
  memcpy(best, cur.positions, sizeof(float) * p.problem_dimensions);
  unsigned seed = 0;
  float best_fitness = cur.fitness(best, &seed, p.problem_dimensions);

  // exit(0);
  //  MAIN COMPUTATION
  for (unsigned int i = 0; i < p.iterations; i++) {
    logical_chunk_size = (p.population_size + c.count - 1) / c.count;
    unsigned int n_local_chunks = assing_logical_chunks(
        cur.start, cur.end, phisical_chunk_size, p.population_size,
        threads - rank_id - 1, logical_chunk_size, local_chunks);
    // printf("%d %d\n", rank_id, n_local_chunks);
    //  1) accumulate
    for (unsigned int j = 0; j < n_local_chunks; j++) {
      // TODO paralelization opportunity if handling correctly random seed
      new_computation_accumulate(&cur, &local_chunks[j], &cur.seed);
      // printf("ACCUMULATED? %f", *cur.positions);
    }

    // 2) send rightmost part to 1 left
    // MPI_Barrier(MPI_COMM_WORLD);
    // sleep(3);
    // RECIVING FROM RIGHT
    if (local_chunks[n_local_chunks - 1].to_shift_right) {
      MPI_Status s;
      int err = 0;
      // SENDING LEFT
      if (local_chunks[0].to_shift_left) {
        printf("1sendrecv %d<-(%d)<-%d\n", rank_id - 1, rank_id, rank_id + 1);
        fflush(stdout);
        err =
            MPI_Sendrecv(&local_chunks[0].comp, 1, computation_status_type,
                         rank_id - 1, 0, &temp_comp, 1, computation_status_type,
                         rank_id + 1, 0, MPI_COMM_WORLD, &s);
        printf("1ok sendrecv %d<-(%d)<-%d\n", rank_id - 1, rank_id,
               rank_id + 1);
        fflush(stdout);
      } else {
        printf("1recv (%d)<-%d\n", rank_id, rank_id + 1);
        fflush(stdout);
        err = MPI_Recv(&temp_comp, 1, computation_status_type, rank_id + 1, 0,
                       MPI_COMM_WORLD, &s);
        printf("1ok sendrecv (%d)<-%d\n", rank_id, rank_id + 1);
        fflush(stdout);
      }
      if (err != MPI_SUCCESS) {
        char err_string[MPI_MAX_ERROR_STRING];
        int err_len;
        MPI_Error_string(err, err_string, &err_len);
        // printf("Rank %d: MPI_Recv failed with error: %s\n", rank_id,
        //        err_string);
      }
      computation_status_merge(&local_chunks[n_local_chunks - 1].comp,
                               &temp_comp, p.problem_dimensions);
    } else if (local_chunks[0].to_shift_left) {
      // SENDING LEFT
      // printf("1send %d<-(%d)\n", rank_id - 1, rank_id);
      // fflush(stdout);
      MPI_Send(&local_chunks[0].comp, 1, computation_status_type, rank_id - 1,
               0, MPI_COMM_WORLD);
      // printf("1ok send %d<-(%d)\n", rank_id - 1, rank_id);
      // fflush(stdout);
    }
    /*MPI_Barrier(MPI_COMM_WORLD);
    sleep(1);
    if (rank_id == 0) {
      sleep(1);
      printf("BARRIER1#########################\n");
    }*/
    MPI_Barrier(MPI_COMM_WORLD);
    // 3) butterfly
    comunicate(&local_chunks[n_local_chunks - 1], computation_status_type,
               phisical_chunk_size, rank_id, threads, p.population_size,
               p.problem_dimensions);
    // 4) rightize

    /*MPI_Barrier(MPI_COMM_WORLD);
    sleep(1);
    if (rank_id == 0) {
      sleep(1);
      printf("BARRIER2#########################\n");
    }*/
    MPI_Barrier(MPI_COMM_WORLD);

    // RECEIVING FROM LEFT
    if (local_chunks[0].to_shift_left) {
      MPI_Status s;
      int err = 0;
      // SENDING RIGHT
      if (local_chunks[n_local_chunks - 1].to_shift_right) {
        // printf("sendrecv %d->(%d)->%d\n", rank_id - 1, rank_id, rank_id + 1);
        // fflush(stdout);
        err = MPI_Sendrecv(&local_chunks[n_local_chunks - 1].comp, 1,
                           computation_status_type, rank_id + 1, 0, &temp_comp,
                           1, computation_status_type, rank_id - 1, 0,
                           MPI_COMM_WORLD, &s);
        // printf("ok sendrecv %d<-(%d)<-%d, %d\n", rank_id - 1, rank_id,
        //        rank_id + 1, s.MPI_ERROR);
      } else {
        // printf("recv %d->(%d)\n", rank_id - 1, rank_id);
        // fflush(stdout);
        err = MPI_Recv(&temp_comp, 1, computation_status_type, rank_id - 1, 0,
                       MPI_COMM_WORLD, &s);
        // printf("ok recv %d->(%d) %d\n", rank_id - 1, rank_id, s.MPI_ERROR);
      }
      if (err != MPI_SUCCESS) {
        char err_string[MPI_MAX_ERROR_STRING];
        int err_len;
        MPI_Error_string(err, err_string, &err_len);
        printf("Rank %d: MPI_Recv failed with error: %s\n", rank_id,
               err_string);
      }
      memcpy(&local_chunks[0].comp, &temp_comp, sizeof(ComputationStatus));
      // computation_status_merge(&local_chunks[0].comp, &temp_comp,
      // p.problem_dimensions);
    } else if (local_chunks[n_local_chunks - 1].to_shift_right) {
      // SENDING RIGHT
      // printf("send (%d)->%d\n", rank_id, rank_id + 1);
      // fflush(stdout);
      MPI_Send(&local_chunks[n_local_chunks - 1].comp, 1,
               computation_status_type, rank_id + 1, 0, MPI_COMM_WORLD);
      // printf("ok send (%d)->%d\n", rank_id, rank_id + 1);
    }
    // TODO remove sanity check
    for (unsigned j = 0; j < n_local_chunks; j++) {
      if (local_chunks[j].comp.n !=
          local_chunks[j].end - local_chunks[j].start) {
        printf("ERROR [%d %d]: %d %d %f\n", local_chunks[j].start,
               local_chunks[j].end, local_chunks[j].comp.n,
               local_chunks[j].end - local_chunks[j].start,
               *local_chunks[j].comp.cumulated_pos);
        fflush(stdout);
        sleep(1);
        assert(local_chunks[j].comp.n ==
               local_chunks[j].end - local_chunks[j].start);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    /*
        if (rank_id == 0) {
          sleep(3);
          printf("BARRIER#########################\n");
        }

        fflush(stdout);
        sleep(3);
        MPI_Barrier(MPI_COMM_WORLD);*/

    // 5) compute

    for (unsigned int k = 0; k < n_local_chunks; k++) {
      scalar_prod_assign(local_chunks[k].comp.cumulated_speeds,
                         1.0 / (float)local_chunks[k].comp.n,
                         p.problem_dimensions);
      // TODO fix random
      inner_dragonfly_step(&cur, local_chunks[k], cur.seed);
      if (local_chunks[k].comp.next_food_fitness > best_fitness) {
        best_fitness = local_chunks[k].comp.next_food_fitness;
        memcpy(best, local_chunks[k].comp.next_food,
               sizeof(float) * p.problem_dimensions);
      }
    }
    // 6) update parameters
    update_chunk_size(&c);
  }

  // Clean up allocated memory
  free(local_chunks);
  dragonfly_free(cur);
  // Free the MPI datatype
  MPI_Type_free(&computation_status_type);
  // Return the best solution found (for now, return a valid result)
  printf("FUUUCK %f\n", best_fitness);
  return best;
}

/*// take timing not including IO
float *dragonfly_compute(Parameters p, Weights w, Fitness fitness, unsigned
int threads, unsigned int rank_id, float space_size, unsigned int srand) {

  if(p.n_chunks>MAX_CHUNKS){
    fprintf(stderr, "impossible to compute with %d chunks, recompile with a
bigger MAX_CHUNKS\n", p.n_chunks); exit(-1);
  }
  if(p.n_chunks==0){
    fprintf(stderr, "impossible to compute with 0 chunks\n");
    exit(-1);
  }

  #ifndef USE_MPI
    if(threads>1){
      fprintf(stderr, "threads>1, recompile with mpi");
      exit(-1);
    }
  #endif
  if (p.threads_per_process>1){

  }
  // compute start and end rank
  unsigned int start_chunk = rank_id*(p.n_chunks/threads) + min(rank_id,
p.n_chunks%threads); unsigned int current_chunks = p.n_chunks/threads +
(rank_id<(p.n_chunks%threads)); unsigned int end_chunk = start_chunk +
current_chunks;


  #ifdef USE_MPI
    MPI_Datatype message_type, computation_status_type;
    Build_mpi_type_computation_status(&computation_status_type);
    build_mpi_type_message(&message_type, &computation_status_type);
  #endif

  Dragonfly *d = malloc(sizeof(Dragonfly) * (current_chunks));

  // allocate problem
  for (unsigned int i = 0; i < current_chunks; i++) {
    unsigned int cur_n = p.population_size/p.n_chunks +
(i<(p.population_size%p.n_chunks)); d[i] =
dragonfly_new(p.problem_dimensions, cur_n, p.iterations, space_size, w,
fitness, i + start_chunk+ srand); dragonfly_alloc(&d[i]);
  }


  float *best = malloc(sizeof(float) * p.problem_dimensions);
  memcpy(best, d->positions, p.problem_dimensions * sizeof(float));
  float best_fitness = d->fitness(best, &d->seed, p.problem_dimensions);

  unsigned int joint_chunks = 1;
  int log_chunks = 0;
  int tmp_chunks = p.n_chunks;
  while (tmp_chunks > 1) {
    tmp_chunks /= 2;
    log_chunks++;
  }

  unsigned int update_chunk_steps = (p.iterations + log_chunks) /
(log_chunks + 1); Message message;

  // for each iteration
  for (unsigned int i = 0; i < p.iterations; i++) {
    if (i != 0 && i % update_chunk_steps == 0) {
      joint_chunks *= 2;
    }

    // compute avarage speed and positions
    for(unsigned int j=0; j<current_chunks; j++){
      computation_accumulate(&message.status[j+start_chunk], &d[j], best,
&best_fitness, p.threads_per_process);
    }
    message.start_chunk=start_chunk;
    message.end_chunk=end_chunk;

    //send current chunks to others, and retrive collective status (if
USE_MPI is defined) #ifdef USE_MPI message_broadcast(&message, rank_id,
threads, &message_type); #endif
    //prepare and compute
    for(unsigned int j=start_chunk-start_chunk%joint_chunks; j<end_chunk;
j+=joint_chunks){ for(unsigned int k=1; k<joint_chunks; k++){
        if(j+k<p.n_chunks){
          computation_status_merge(&message.status[j], &message.status[j+k],
p.problem_dimensions);
        }
      }
      scalar_prod_assign(message.status[j].cumulated_speeds,
                         1.0 / (float)message.status[j].n,
p.problem_dimensions); for(unsigned int k=0; k<joint_chunks; k++){
        if(start_chunk<=j+k && j+k<end_chunk){
          dragonfly_compute_step(&d[j+k-start_chunk],
message.status[j].cumulated_speeds, message.status[j].cumulated_pos,
message.status[j].next_food, message.status[j].next_enemy,
message.status[j].n, p.threads_per_process);
        }
      }
    }
  }
  // check last iteration
  // compute avarage speed and positions
  for(unsigned int j=0; j<current_chunks; j++){
      computation_accumulate(&message.status[j+start_chunk], &d[j], best,
&best_fitness, p.threads_per_process);
    }
  message.start_chunk=start_chunk;
  message.end_chunk=end_chunk;

  if(message.status[start_chunk].next_food_fitness< best_fitness) {
    memcpy(message.status[start_chunk].next_food, best,
p.problem_dimensions*sizeof(float));
    message.status[start_chunk].next_food_fitness = best_fitness;
  }

  //send current chunks to others, and retrive collective status (if USE_MPI
is defined) #ifdef USE_MPI message_broadcast(&message, rank_id, threads,
&message_type); #endif

  for(unsigned int i=0; i<p.n_chunks; i++){
    computation_status_merge(&message.status[0], &message.status[i],
p.problem_dimensions);
  }

  memcpy(best, message.status[start_chunk].next_food,
p.problem_dimensions*sizeof(float)); for(unsigned int i=0; i<current_chunks;
i++){ dragonfly_free(d[i]);
  }
  free(d);
  //memcpy(best, message.status[0].next_food, p.dim*sizeof(float));
  #ifdef USE_MPI
  MPI_Type_free(&message_type);
  MPI_Type_free(&computation_status_type);
  #endif
  return best;
}*/

ChunkSize new_chunk_size(unsigned int start_count, unsigned int end_count,
                         unsigned int steps) {
  ChunkSize r = {
      .start_count = start_count,
      .end_count = end_count,
      .current_step = 0,
      .total_steps = steps,
      .count = start_count,
  };
  return r;
}

void update_chunk_size(ChunkSize *c) {
  c->current_step++;
  float temp = powf((float)c->end_count / (float)c->start_count,
                    (float)c->current_step / (float)c->total_steps) *
               (float)c->start_count;
  c->count = (unsigned int)(round(temp));
  printf("%d %d %f\n", c->current_step, c->count, temp);
}