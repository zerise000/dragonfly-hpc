#include "dragonfly-common.h"
#include "stdlib.h"
#include "string.h"
#include "utils.h"
#include <time.h>
#include <float.h>

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include <mpi.h>
#include <unistd.h>

// TODO remove signature
void new_computation_accumulate(Dragonfly *d, LogicalChunk *current_chunk,
                                unsigned int *seed, unsigned int nr_threads);
// #endif

// n, chunks, iterations, dim
Parameters parameter_parse(int argc, char *argv[]) {
  Parameters p;
  if (argc != 6) {
    fprintf(stderr, "invalid parameter count: expected population_size, "
                    "starting_chunk_count, iterations, problem_dimensions, "
                    "threads_per_process\n");
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
                        void *fitness_data, unsigned fitness_data_size,
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
      .fitness_data = fitness_data,
      .fitness_data_size = fitness_data_size,
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
  for (unsigned int i = 0; i < required_steps; i++) {

    unsigned partner = (inner_rank_id ^ (1 << i)) + start;

    if (start <= partner && partner < end) {
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
      MPI_Sendrecv(&cur->comp, 1, type, partner, 1, &tmp, 1, type, partner, 1,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if (cur->comp.n < tmp.n) {
        memcpy(&cur->comp, &tmp, sizeof(ComputationStatus));
      }
    }
  }
}

void inner_dragonfly_step(Dragonfly *d, LogicalChunk chunk,
                          unsigned int random_seed) {

  unsigned int dimensions = d->dim;
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

      cur_speed[i] = min(cur_speed[i], d->space_size);
      cur_speed[i] = max(cur_speed[i], -d->space_size);

      cur_pos[i] += cur_speed[i];

      cur_pos[i] = min(cur_pos[i], d->space_size);
      cur_pos[i] = max(cur_pos[i], -d->space_size);
    }
  }
}

float *dragonfly_compute(Parameters p, Weights w, ChunkSize c, Fitness fitness,
                         void *fitness_data, unsigned fitness_data_size,
                         unsigned int process_count, unsigned int rank_id,
                         float space_size, unsigned int srand) {

  if (MESSAGE_SIZE < p.problem_dimensions) {
    fprintf(stderr,
            "impossible to compute with %d dimensions, recompile with a bigger "
            "MESSAGE_SIZE\n",
            p.problem_dimensions);
    exit(-1);
  }
#ifndef USE_MPI
  if (process_count > 1) {
    fprintf(stderr, "impossible to compute with multiple process without mpi, "
                    "recompile with USE_MPI\n");
    exit(-1);
  }
#endif
  // TODO remove temporary asserts
  assert(p.population_size >= process_count);
  assert(p.population_size >= c.start_count);
  // compute start and end phisical_thread
  unsigned int phisical_chunk_size = p.population_size / process_count;
  unsigned int phisical_start = phisical_chunk_size * rank_id;
  unsigned int phisical_end = rank_id == process_count - 1
                                  ? p.population_size
                                  : phisical_start + phisical_chunk_size;
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
  void *computation_status_type = NULL;
  if (process_count > 1) {
    computation_status_type = malloc(sizeof(MPI_Datatype));
    Build_mpi_type_computation_status(computation_status_type);
  }

  LogicalChunk *local_chunks =
      malloc(sizeof(LogicalChunk) * maximum_logical_chunks);

  // Initialize the allocated memory to avoid uninitialized data
  memset(local_chunks, 0, sizeof(LogicalChunk) * maximum_logical_chunks);

  Dragonfly cur = dragonfly_new(
      p.problem_dimensions, phisical_start, phisical_end, p.iterations,
      space_size, w, fitness, fitness_data, fitness_data_size, rank_id + srand);

  dragonfly_alloc(&cur);
  ComputationStatus temp_comp;
  float *best = malloc(sizeof(float) * p.problem_dimensions);
  memcpy(best, cur.positions, sizeof(float) * p.problem_dimensions);
  unsigned seed = 0;
  float best_fitness =
      cur.fitness(best, &seed, p.problem_dimensions, fitness_data);
  // exit(0);
  //  MAIN COMPUTATION
  for (unsigned int i = 0; i < p.iterations; i++) {
    logical_chunk_size = (p.population_size + c.count - 1) / c.count;
    unsigned int n_local_chunks = assing_logical_chunks(
        cur.start, cur.end, phisical_chunk_size, p.population_size,
        process_count - rank_id - 1, logical_chunk_size, local_chunks);
    //  1) accumulate

    //accumulate time with wall-clock time (not CPU time)

    if (p.threads_per_process > 1) {
#pragma omp parallel for num_threads(p.threads_per_process)
      for (unsigned int j = 0; j < n_local_chunks; j++) {

        new_computation_accumulate(&cur, &local_chunks[j], &cur.seed,
                                   p.threads_per_process);
      }
    } else

    {
      for (unsigned int j = 0; j < n_local_chunks; j++) {
        new_computation_accumulate(&cur, &local_chunks[j], &cur.seed, 1);
      }
    }

    if (process_count > 1) {

      // 2) send rightmost part to 1 left
      // RECIVING FROM RIGHT
      if (local_chunks[n_local_chunks - 1].to_shift_right) {
        MPI_Status s;
        int err = 0;
        // SENDING LEFT
        if (local_chunks[0].to_shift_left) {
          err = MPI_Sendrecv(&local_chunks[0].comp, 1,
                             *(MPI_Datatype *)computation_status_type,
                             rank_id - 1, 0, &temp_comp, 1,
                             *(MPI_Datatype *)computation_status_type,
                             rank_id + 1, 0, MPI_COMM_WORLD, &s);
        } else {
          err =
              MPI_Recv(&temp_comp, 1, *(MPI_Datatype *)computation_status_type,
                       rank_id + 1, 0, MPI_COMM_WORLD, &s);
        }
        if (err != MPI_SUCCESS) {
          char err_string[MPI_MAX_ERROR_STRING];
          int err_len;
          MPI_Error_string(err, err_string, &err_len);
        }
        computation_status_merge(&local_chunks[n_local_chunks - 1].comp,
                                 &temp_comp, p.problem_dimensions);
      } else if (local_chunks[0].to_shift_left) {
        // SENDING LEFT
        MPI_Send(&local_chunks[0].comp, 1,
                 *(MPI_Datatype *)computation_status_type, rank_id - 1, 0,
                 MPI_COMM_WORLD);
      }
      MPI_Barrier(MPI_COMM_WORLD);

      // 3) butterfly
      comunicate(&local_chunks[n_local_chunks - 1],
                 *(MPI_Datatype *)computation_status_type, phisical_chunk_size,
                 rank_id, process_count, p.population_size,
                 p.problem_dimensions);

      // 4) rightize
      MPI_Barrier(MPI_COMM_WORLD);

      // RECEIVING FROM LEFT
      if (local_chunks[0].to_shift_left) {
        MPI_Status s;
        int err = 0;
        // SENDING RIGHT
        if (local_chunks[n_local_chunks - 1].to_shift_right) {
          err = MPI_Sendrecv(&local_chunks[n_local_chunks - 1].comp, 1,
                             *(MPI_Datatype *)computation_status_type,
                             rank_id + 1, 0, &temp_comp, 1,
                             *(MPI_Datatype *)computation_status_type,
                             rank_id - 1, 0, MPI_COMM_WORLD, &s);
        } else {
          err =
              MPI_Recv(&temp_comp, 1, *(MPI_Datatype *)computation_status_type,
                       rank_id - 1, 0, MPI_COMM_WORLD, &s);
        }
        if (err != MPI_SUCCESS) {
          char err_string[MPI_MAX_ERROR_STRING];
          int err_len;
          MPI_Error_string(err, err_string, &err_len);
          printf("Rank %d: MPI_Recv failed with error: %s\n", rank_id,
                 err_string);
        }
        memcpy(&local_chunks[0].comp, &temp_comp, sizeof(ComputationStatus));
      } else if (local_chunks[n_local_chunks - 1].to_shift_right) {
        // SENDING RIGHT
        MPI_Send(&local_chunks[n_local_chunks - 1].comp, 1,
                 *(MPI_Datatype *)computation_status_type, rank_id + 1, 0,
                 MPI_COMM_WORLD);
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
    }
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
  if (process_count > 1) {
    MPI_Type_free(computation_status_type);
    free(computation_status_type);
  }

  // Return the best solution found (for now, return a valid result)
  return best;
}

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
}

void update_status(Dragonfly *d, LogicalChunk *current_chunk,
                   float next_food_fitness, float next_enemy_fitness,
                   unsigned int *indexes, unsigned int start, unsigned int end,
                   unsigned int dim) {

  memcpy(current_chunk->comp.next_food, d->positions + indexes[0] * dim,
         sizeof(float) * dim);
  memcpy(current_chunk->comp.next_enemy, d->positions + indexes[1] * dim,
         sizeof(float) * dim);
  current_chunk->comp.next_enemy_fitness = next_enemy_fitness;
  current_chunk->comp.next_food_fitness = next_food_fitness;
  current_chunk->comp.n = end - start;
}

void inner_new_computation_accumulate(Dragonfly *d, LogicalChunk *current_chunk,
                                      unsigned int *seed, unsigned start,
                                      unsigned end, void *fitness_data) {
  unsigned dim = d->dim;
  float *cumulated_pos = current_chunk->comp.cumulated_pos;
  float *cumulated_speed = current_chunk->comp.cumulated_speeds;
  zeroed(cumulated_pos, dim);
  zeroed(cumulated_speed, dim);

  float next_enemy_fitness = FLT_MAX;
  float next_food_fitness = -FLT_MAX;
  unsigned int indexes[2] = {0, 0};
  end = end - d->start;

  for (unsigned int k = start - d->start; k < end; k++) {
    float *iter_pos = d->positions + dim * k;
    float *iter_speed = d->speeds + dim * k;
    sum_assign(cumulated_pos, iter_pos, dim);
    sum_assign(cumulated_speed, iter_speed, dim);

    float fitness = d->fitness(iter_pos, seed, dim, fitness_data);

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
void new_computation_accumulate(Dragonfly *d, LogicalChunk *current_chunk,
                                unsigned int *seed, unsigned int nr_threads) {
  unsigned start = max(current_chunk->start, d->start);
  unsigned end = min(current_chunk->end, d->end);

  unsigned dim = d->dim;
  if (nr_threads == 1 || nr_threads > end - start) {
    inner_new_computation_accumulate(d, current_chunk, seed, start, end,
                                     d->fitness_data);
    return;
  }
#ifdef USE_OPENMP
  LogicalChunk *temp_chunks =
      (LogicalChunk *)malloc(sizeof(LogicalChunk) * nr_threads);
  unsigned int elems_per_thread = (end - start) / nr_threads;

#pragma omp parallel for num_threads(nr_threads)
for(unsigned int thread_id=0; thread_id<nr_threads; thread_id++)
  {
    //unsigned int thread_id = omp_get_thread_num();
    void *tmp_fitness_data = NULL;
    if (d->fitness_data_size != 0) {
      tmp_fitness_data = malloc(d->fitness_data_size);
      memcpy(tmp_fitness_data, d->fitness_data, d->fitness_data_size);
    }

    memcpy(&temp_chunks[thread_id], current_chunk, sizeof(LogicalChunk));

    unsigned int thread_start = start + (thread_id * elems_per_thread);
    unsigned int thread_end = thread_id == (nr_threads - 1)
                                  ? end
                                  : min(thread_start + elems_per_thread, end);
    unsigned int local_seed = 0;
    inner_new_computation_accumulate(d, &temp_chunks[thread_id], &local_seed,
                                     thread_start, thread_end,
                                     tmp_fitness_data);
    if (d->fitness_data_size != 0) {
      free(tmp_fitness_data);
    }
  }
  memcpy(current_chunk, &temp_chunks[0], sizeof(LogicalChunk));
  for (unsigned int i = 1; i < nr_threads; i++) {
    computation_status_merge(&current_chunk->comp,
                             &temp_chunks[i].comp, dim);
  }
  
  free(temp_chunks);
#else
  fprintf(stderr,
          "impossible to compute with %d threads, recompile with USE_OPENMP "
          "flag \n",
          nr_threads);
  exit(-1);
#endif
}
