#ifndef DA_COMMON
#define DA_COMMON

typedef struct {
  float sl[2], al[2], cl[2], fl[2], el[2], wl[2], ll[2], max_speedl[2];
  float st, at, ct, ft, et, wt, lt, max_speedt;
  float s, a, c, f, e, w, l, max_speed;
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
  unsigned int N, chunks, chunks_id;

  // buffers TODO (keep them?)
  float *positions, *speeds;

  // tmp buffers (in order to not allocate and deallocate memory)
  float *S, *A, *C, *F, *E, *W, *delta_pos, *levy;
} Dragonfly;

Dragonfly dragonfly_new(unsigned int dimensions, unsigned int N,
                        unsigned int chunks, unsigned int chunk_id,
                        unsigned int iterations, float space_size,
                        Weights weights, Fitness fitness);
void dragonfly_alloc(Dragonfly *d);
void dragonfly_free(Dragonfly d);
void dragonfly_compute_step(Dragonfly *d, float *average_speed,
                            float *cumulated_pos, float *food, float *enemy,
                            unsigned int N);

typedef struct {
  float cumulated_pos[50];
  float cumulated_speeds[50];

  float next_enemy[50];
  float next_enemy_fitness;
  float next_food[50];
  float next_food_fitness;

  unsigned int n;
} Message;

typedef struct {
  unsigned int n, chunks, iterations, dim;

} Parameters;

Parameters parameter_parse(int argc, char *argv[]);

void message_broadcast(Message *my_value, unsigned int i, unsigned int incr,
                       void *data, int dim,
                       void (*raw_sendrecv)(Message *, unsigned int, Message *,
                                            unsigned int, void *));

void message_acumulate(Message *message, Dragonfly *d, float *best,
                       float *best_fitness);

#endif