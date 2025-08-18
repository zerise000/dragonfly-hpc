#ifndef DA_COMMON
#define DA_COMMON
#include <math.h>
#define MESSAGE_SIZE 16
#define MAX_CHUNKS 4048


// struct used to store the progression of the dragonfly weights s, a, c, f, e, w, l (linear)

typedef struct {
  float sl[2], al[2], cl[2], fl[2], el[2], wl[2], ll[2];
  float st, at, ct, ft, et, wt, lt;
  float s, a, c, f, e, w, l;


} Weights;

// also it stores the logical_chunk_count progression (geometric)
typedef struct {
    unsigned start_count, end_count;
  unsigned int current_step;
  unsigned int total_steps;
  unsigned count;

} ChunkSize;

ChunkSize new_chunk_size(unsigned int start_count, unsigned int end_count, unsigned int steps);

void update_chunk_size(ChunkSize* c);

// Newtype for Fitness functions, they should take the input vector, one int as random number generator seed, and the dimension count
typedef float (*Fitness)(float *, unsigned int*, unsigned int);

// function used to compute the step for the linear progression of the weights.
void weights_compute_steps(Weights *w, unsigned int steps);

// compute one step forward for the weights
void weights_step(Weights *w);


typedef struct {
  // dimensions of the problem
  unsigned int dim;
  // size of the search space
  float space_size;
  // fitness function
  Fitness fitness;
  //adaptive weights
  Weights w;
  
  // start and end of elements used by the current thread
  unsigned int start, end;
  
  // local_buffers the size is local_end-local_start elements
  float *positions, *speeds;

  // tmp buffers (in order to not allocate and deallocate memory)
  float *S, *A, *C, *F, *E, *W, *delta_pos, *levy;
  
  //random seed
  unsigned int seed;
} Dragonfly;


Dragonfly dragonfly_new(unsigned int dimensions, unsigned int start, unsigned int end,
                        unsigned int iterations, float space_size,
                        Weights weights,
                        Fitness fitness, unsigned int random_seed);
                        
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


typedef struct {
  unsigned int population_size, starting_chunk_count, iterations, problem_dimensions, threads_per_process;
} Parameters;

Parameters parameter_parse(int argc, char *argv[]);



//void message_broadcast(Message *message, unsigned int index, int n, MPI_Datatype *data_type);
void computation_status_merge(ComputationStatus *out, ComputationStatus *in, unsigned int dim);
float *dragonfly_compute(Parameters p, Weights w, ChunkSize c, Fitness fitness, unsigned int threads, unsigned int rank_id, float space_size, unsigned int srand);
#endif