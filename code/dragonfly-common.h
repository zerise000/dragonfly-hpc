#ifndef DA_COMMON
#define DA_COMMON

typedef struct {
  float sl[2], al[2], cl[2], fl[2], el[2], wl[2];
  float st, at, ct, ft, et, wt;
  float s, a, c, f, e, w;
} Weights;

void weights_compute_steps(Weights *w, unsigned int steps);
void weights_step(Weights *w);

typedef struct {
  // dimensions of the problem
  unsigned int dim;
  float space_size;
  float (*fitness)(float *, unsigned int);
  Weights w;
  // for how many dragonflies? problem definition
  // N is the chunk_size, not the problem size
  unsigned int N, iter, chunks, chunks_id;

  // buffers TODO (keep them?)
  float *positions, *speeds;

  // tmp buffers (in order to not allocate and deallocate memory)
  float *S, *A, *C, *F, *E, *W, *delta_pos;
} Dragonfly;

Dragonfly dragonfly_new(unsigned int dimensions, unsigned int N, unsigned int chunks, unsigned int chunk_id,
                        unsigned int iterations, float space_size,
                        Weights weights,
                        float (*fitness)(float *, unsigned int));
void dragonfly_alloc(Dragonfly *d);
void dragonfly_free(Dragonfly d);
void dragonfly_compute_step(Dragonfly *d, float *average_speed,
                            float *cumulated_pos, float * food, float * enemy);

typedef struct {
  float cumulated_pos[50];
  float cumulated_speeds[50];

  float next_enemy[50];
  float next_enemy_fitness;
  float next_food[50];
  float next_food_fitness;

  unsigned int n;
} Message;

void message_broadcast(Message *my_value, unsigned int i, unsigned int incr,
                       void *data, int dim,
                       void (*raw_sendrecv)(Message *, unsigned int, Message *,
                                            unsigned int, void *));
                                            
#endif