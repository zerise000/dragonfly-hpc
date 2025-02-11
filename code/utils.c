#include <math.h>
#include <stdlib.h>

#include "dragonfly-common.h"
#include <string.h>
/*
void my_free(void* mem, const char *file, int line, const char *func)
{

    free(mem);
    printf ("Freed = %s, %i, %s,\n", file, line, func);


}
void* my_malloc(size_t size, const char *file, int line, const char *func)
{

    void *p = malloc(size);
    printf ("Allocated = %s, %i, %s, %p[%li]\n", file, line, func, p, size);

    return p;
}*/
#include "utils.h"

float *init_array(unsigned int dimensions, float range_max,
                  unsigned int *seed) {
  float *arr = malloc(dimensions * sizeof(float));
  for (unsigned int i = 0; i < dimensions; i++) {
    arr[i] = RAND_FLOAT(range_max, seed);
  }
  return arr;
}

void zeroed(float *dest, unsigned int size) {
  for (unsigned int i = 0; i < size; i++) {
    dest[i] = 0.0;
  }
}

void sum_assign(float *dest, float *source, unsigned int size) {
  for (unsigned int i = 0; i < size; i++) {
    dest[i] += source[i];
  }
}

void sub_assign(float *dest, float *source, unsigned int size) {
  for (unsigned int i = 0; i < size; i++) {
    dest[i] -= source[i];
  }
}

void scalar_prod_assign(float *dest, float val, unsigned int size) {
  for (unsigned int i = 0; i < size; i++) {
    dest[i] *= val;
  }
}

float dot_product(float *a, float *b, unsigned int size) {
  float ret = 0.0;
  for (unsigned int i = 0; i < size; i++) {
    ret += a[i] * b[i];
  }
  return ret;
}

float length(float *inp, unsigned int size) {
  float ret = dot_product(inp, inp, size);

  return sqrt(ret);
}

// Gram-Schmidt orthonormal matrix
void init_matrix(float *inp, float range_max, unsigned int dim,
                 unsigned int *seed) {
  float *tmp = malloc(sizeof(float) * dim);
  for (unsigned int row = 0; row < dim; row++) {
    for (unsigned int col = 0; col < dim; col++) {
      inp[row * dim + col] = RAND_FLOAT(range_max, seed);
    }

    // make it perpendicular
    for (unsigned int prow = 0; prow < row; prow++) {
      float cur = dot_product(inp + row * dim, inp + prow * dim, dim);
      cur /= dot_product(inp + row * dim, inp + prow * dim, dim);
      memcpy(tmp, inp + prow * dim, dim * sizeof(float));
      scalar_prod_assign(tmp, cur, dim);
      sub_assign(inp + row * dim, tmp, dim);
    }

    // normalize
    float l = length(inp + row * dim, dim);
    scalar_prod_assign(inp + row * dim, 1.0 / l, dim);
  }
  free(tmp);
}

void matrix_times_vector(float *out, float *matrix, float *inp,
                         unsigned int dim) {
  for (unsigned int row = 0; row < dim; row++) {
    out[row] = dot_product(matrix + row * dim, inp, dim);
  }
}

/*
float sigma(float beta) {
  return tgamma(beta + 1) * sin(beta * M_PI / 2) / tgamma((beta + 1) / 2) *
         beta * pow(2, (beta - 1) / 2);
}

float levy() {
  float beta = 4.25;
  return 0.01 * (RAND_FLOAT(1) * sigma(beta) / pow(RAND_FLOAT(1), 1 / beta));
}*/

void brownian_motion(float *inp, unsigned int dim, unsigned int *seed) {

  for (unsigned int i = 0; i < dim; i++) {
    inp[i] = RAND_FLOAT(1.0, seed);
  }
}

float rastrigin_fitness(float *inp, unsigned int *seed, unsigned int dim) {
  (void)(seed);
  float ret = 10.0 * dim;
  for (unsigned int i = 0; i < dim; i++) {
    ret += inp[i] * inp[i] - 10.0 * cos(2 * M_PI * inp[i]);
  }
  return -ret;
}

float sphere_fitness(float *inp, unsigned int *seed, unsigned int dim) {
  (void)(seed);
  float res = 0.0;
  for (unsigned int i = 0; i < dim; i++) {
    res += inp[i] * inp[i];
  }
  return -res;
}
float rosenblock_fitness(float *inp, unsigned int *seed, unsigned int dim) {
  (void)(seed);
  float res = 0.0;
  for (unsigned int i = 0; i < dim - 1; i++) {
    res +=
        100 * pow(inp[i + 1] - inp[i] * inp[i], 2.0) + pow(1.0 - inp[i], 2.0);
  }
  return -res;
}

float *ROTATION;
float *SHIFT;
Fitness FITNESS;
float *TMP;
void init_shifted_fitness(float *tmp, float *rotation, float *shift,
                          Fitness fitness) {
  ROTATION = rotation;
  SHIFT = shift;
  FITNESS = fitness;
  TMP = tmp;
}

float shifted_fitness(float *inp, unsigned int *seed, unsigned int dim) {
  matrix_times_vector(TMP, ROTATION, inp, dim);
  sum_assign(TMP, SHIFT, dim);
  return FITNESS(TMP, seed, dim);
}

#ifdef USE_OPENMP
#include <omp.h>
void dragonfly_compute_step_multithread(Dragonfly *d, float *average_speed,
                                        float *cumulated_pos, float *food,
                                        float *enemy, unsigned int N,
                                        unsigned int NR_THREADS) {
  unsigned int rest = d->N % NR_THREADS;
  unsigned int ratio = d->N / NR_THREADS;
  unsigned int dimensions = d->dim;
  printf("serial rest=%d, ratio=%d, dim=%d\n", rest, ratio, dimensions);
  fflush(stdout);
#pragma omp parallel num_threads(NR_THREADS)
  {

    unsigned int rank = omp_get_thread_num();
    unsigned int base = ratio * rank;
    unsigned int limit = ratio * (rank + 1);

    float S;
    float A;
    float C;
    float F;
    float E;
    float levy;

    float *cur_pos;
    float *cur_speed;

    for (unsigned int j = base; j < limit; j++) {
      cur_pos = d->positions + j * dimensions;
      cur_speed = d->speeds + j * dimensions;

      // compute speed = sSi + aAi + cCi + fFi + eEi + w

      for (unsigned int i = 0; i < dimensions; i++) {
        S = (cumulated_pos[i] / ((float)N)) - cur_pos[i];
        A = average_speed[i];
        C = (cumulated_pos[i] / (float)N) - cur_pos[i];
        F = food[i] - cur_pos[i];
        E = enemy[i] + cur_pos[i];
        levy = RAND_FLOAT(1.0, &d->seed);

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
  if (rest != 0) {
    unsigned int r_base = ratio * NR_THREADS;
    unsigned int r_end = d->N - 1;

    for (unsigned int j = r_base; j <= r_end; j++) {
      float *cur_pos = d->positions + j * dimensions;
      float *cur_speed = d->speeds + j * dimensions;

      // compute speed = sSi + aAi + cCi + fFi + eEi + w

      for (unsigned int i = 0; i < dimensions; i++) {
        float S = (cumulated_pos[i] / ((float)N)) - cur_pos[i];
        float A = average_speed[i];
        float C = (cumulated_pos[i] / (float)N) - cur_pos[i];
        float F = food[i] - cur_pos[i];
        float E = enemy[i] + cur_pos[i];
        float levy = RAND_FLOAT(1.0, &d->seed);

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

  weights_step(&d->w);
  printf("END\n");
}

void computation_accumulate_multithread(ComputationStatus *message,
                                        Dragonfly *d, float *best,
                                        float *best_fitness,
                                        unsigned int NUM_THREADS) {
  unsigned int dim = d->dim;
  unsigned int rest = d->N % NUM_THREADS;
  unsigned int ratio = d->N / NUM_THREADS;
  printf("rest=%d, ratio=%d, dim=%d N=%d\n", rest, ratio, dim, d->N);
  fflush(stdout);
  zeroed(message->cumulated_pos, dim);
  zeroed(message->cumulated_speeds, dim);
  memcpy(message->next_enemy, d->positions, sizeof(float) * dim);
  memcpy(message->next_food, d->positions, sizeof(float) * dim);
  message->next_enemy_fitness = d->fitness(message->next_enemy, &d->seed, dim);
  message->next_food_fitness = message->next_enemy_fitness;
  message->n = 0;
printf("OK\n");
fflush(stdout);
#pragma omp parallel num_threads(NUM_THREADS)
  {
    int rank = omp_get_thread_num();
    unsigned int base = ratio * rank;
    unsigned int limit = ratio * (rank + 1);
    printf("rank=%d, base=%d, limit=%d\n", rank, base, limit);
    fflush(stdout);
    for (unsigned int k = base; k < limit; k++) {
      float *cur_pos = d->positions + dim * k;
      sum_assign(message->cumulated_pos, cur_pos, dim);
      sum_assign(message->cumulated_speeds, d->speeds + dim * k, dim);

      float fitness = d->fitness(cur_pos, &d->seed, dim);
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

  if (rest != 0) {
    unsigned int r_base = ratio * NUM_THREADS;
    unsigned int r_end = d->N - 1;

    for (unsigned int k = r_base; k <= r_end; k++) {
      float *cur_pos = d->positions + dim * k;
      sum_assign(message->cumulated_pos, cur_pos, dim);
      sum_assign(message->cumulated_speeds, d->speeds + dim * k, dim);
      float fitness = d->fitness(cur_pos, &d->seed, dim);
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
}
#endif

void dragonfly_compute_step_serial(Dragonfly *d, float *average_speed,
                                   float *cumulated_pos, float *food,
                                   float *enemy, unsigned int N,
                                   unsigned int NUM_THREADS) {
  (void)(NUM_THREADS);

  unsigned int dim = d->dim;
  // for each dragonfly
  for (unsigned int j = 0; j < d->N; j++) {
    float *cur_pos = d->positions + dim * j;
    float *cur_speed = d->speeds + dim * j;

    // compute separation: Si = -sumall(X-Xi)
    memcpy(d->S, cumulated_pos, sizeof(float) * dim);
    scalar_prod_assign(d->S, 1.0 / (float)N, dim);
    sub_assign(d->S, cur_pos, dim);
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

    // update current pos
    sum_assign(cur_pos, cur_speed, dim);
  }

  // update weights
  weights_step(&d->w);
}

void computation_accumulate_serial(ComputationStatus *status, Dragonfly *d,
                                   float *best, float *best_fitness,
                                   unsigned int NUM_THREADS) {
  (void)(NUM_THREADS);
  unsigned int dim = d->dim;
  zeroed(status->cumulated_pos, dim);
  zeroed(status->cumulated_speeds, dim);
  memcpy(status->next_enemy, d->positions, sizeof(float) * dim);
  memcpy(status->next_food, d->positions, sizeof(float) * dim);
  status->next_enemy_fitness = d->fitness(status->next_enemy, &d->seed, dim);
  status->next_food_fitness = status->next_enemy_fitness;
  status->n = 0;
  for (unsigned int k = 0; k < d->N; k++) {
    float *cur_pos = d->positions + dim * k;
    sum_assign(status->cumulated_pos, cur_pos, dim);
    sum_assign(status->cumulated_speeds, d->speeds + dim * k, dim);
    float fitness = d->fitness(cur_pos, &d->seed, dim);
    if (fitness > status->next_food_fitness) {
      memcpy(status->next_food, cur_pos, sizeof(float) * dim);
      status->next_food_fitness = fitness;
    }
    if (fitness < status->next_enemy_fitness) {
      memcpy(status->next_enemy, cur_pos, sizeof(float) * dim);
      status->next_enemy_fitness = fitness;
    }
    if (fitness > *best_fitness) {
      memcpy(best, cur_pos, sizeof(float) * dim);
      *best_fitness = fitness;
    }
    status->n += 1;
  }
}

void computation_accumulate(ComputationStatus *status, Dragonfly *d,
                            float *best, float *best_fitness,
                            unsigned int NUM_THREADS) {
  if (NUM_THREADS == 1) {
    computation_accumulate_serial(status, d, best, best_fitness, NUM_THREADS);
  } else {
    #ifdef USE_OPENMP
        computation_accumulate_multithread(status, d, best, best_fitness,
                                          NUM_THREADS);
    #else

        fprintf(
            stderr,
            "Cannot call computation accumulate serial with more than 1 threads\n");
        exit(1);

    #endif
  }
}

void dragonfly_compute_step(Dragonfly *d, float *average_speed,
                            float *cumulated_pos, float *food, float *enemy,
                            unsigned int N, unsigned int NUM_THREADS) {
  if (NUM_THREADS == 1) {
    dragonfly_compute_step_serial(d, average_speed, cumulated_pos, food, enemy,
                                  N, NUM_THREADS);
  } else {
    #ifdef USE_OPENMP
    printf("Using %d threads\n", NUM_THREADS);
        dragonfly_compute_step_multithread(d, average_speed, cumulated_pos, food,
                                          enemy, N, NUM_THREADS);
    #else
        fprintf(stderr, "Cannot call dragonfly compute steps serial with more than "
                        "1 threads\n");
        exit(1);

    #endif
  }
}