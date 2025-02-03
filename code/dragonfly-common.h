#ifndef DA_COMMON
#define DA_COMMON
#include <mpi.h>
#define MESSAGE_SIZE 50
#define MAX_CHUNKS 1024

typedef struct {
  float sl[2], al[2], cl[2], fl[2], el[2], wl[2], ll[2];
  float st, at, ct, ft, et, wt, lt;
  float s, a, c, f, e, w, l;
} Weights;

typedef float (*Fitness)(float *, unsigned int);

void weights_compute_steps(Weights *w, unsigned int steps);
void weights_step(Weights *w);

typedef struct {
  // dimensions of the problem
  unsigned int dim;
  float space_size;
  Fitness fitness;
  Weights w;
  
  // for how many dragonflies? problem definition
  // N is the chunk_size, not the problem size
  unsigned int N;

  // buffers TODO (keep them?)
  float *positions, *speeds;

  // tmp buffers (in order to not allocate and deallocate memory)
  float *S, *A, *C, *F, *E, *W, *delta_pos, *levy;

  unsigned int seed;
} Dragonfly;

Dragonfly dragonfly_new(unsigned int dimensions, unsigned int N,
                        unsigned int iterations, float space_size,
                        Weights weights,
                        float (*fitness)(float *, unsigned int), unsigned int random_seed);
void dragonfly_alloc(Dragonfly *d);
void dragonfly_free(Dragonfly d);



typedef struct {
  float cumulated_pos[MESSAGE_SIZE];
  float cumulated_speeds[MESSAGE_SIZE];

  float next_enemy[MESSAGE_SIZE];
  float next_enemy_fitness;
  float next_food[MESSAGE_SIZE];
  float next_food_fitness;

  unsigned int n;
} ComputationStatus;


typedef struct{
  unsigned int start_chunk, end_chunk;
  ComputationStatus status[MAX_CHUNKS];
} Message;


typedef struct {
  unsigned int n, chunks, iterations, dim;
} Parameters;

Parameters parameter_parse(int argc, char *argv[]);

void message_broadcast(Message *message, unsigned int index, int n, MPI_Datatype *data_type);
void computation_status_merge(ComputationStatus *out, ComputationStatus *in, unsigned int dim);

#endif