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
  unsigned int N, iter, chunks, chunks_id;

  // buffers TODO (keep them?)
  float *positions, *speeds, *food, *next_food, *enemy, *next_enemy;
  float next_food_fitness, next_enemy_fitness;

  // tmp buffers (in order to not allocate and deallocate memory)
  float *cumulated_pos, *average_speed, *S, *A, *C, *F, *E, *W, *delta_pos;
} Dragonfly;

Dragonfly dragonfly_new(unsigned int dimensions, unsigned int N,
                        unsigned int iterations, float space_size,
                        Weights weights,
                        float (*fitness)(float *, unsigned int));
void dragonfly_alloc(Dragonfly *d);
void dragonfly_free(Dragonfly d);
void dragonfly_compute_step(Dragonfly *d, float *average_speed,
                            float *cumulated_pos);

#endif