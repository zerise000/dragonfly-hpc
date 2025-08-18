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

#endif

void inner_dragonfly_step(Dragonfly *d, float *average_speed,
                          float *cumulated_pos, float *food, float *enemy,
                          unsigned int N, unsigned int base, unsigned int limit,
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

  for (unsigned int j = base; j < limit; j++) {
    unsigned random = random_seed + j;
    cur_pos = d->positions + j * dimensions;
    cur_speed = d->speeds + j * dimensions;

    // compute speed = sSi + aAi + cCi + fFi + eEi + w

   for (unsigned int i = 0; i < dimensions; i++) {
      S = (cumulated_pos[i] / ((float)N)) - cur_pos[i];
      A = average_speed[i];
      C = (cumulated_pos[i] / (float)N) - cur_pos[i];
      F = food[i] - cur_pos[i];
      E = enemy[i] + cur_pos[i];
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
/*
void dragonfly_compute_step(Dragonfly *d, float *average_speed,
                            float *cumulated_pos, float *food, float *enemy,
                            unsigned int N, unsigned int NR_THREADS) {
  unsigned int base_random = rand_r(&d->seed);
  //printf("base_random=%d\n", base_random);
  // if more than 1 thread
  unsigned int rest = d->local_n % NR_THREADS;
  unsigned int ratio = d->local_n / NR_THREADS;
  if (NR_THREADS == 1) {
    inner_dragonfly_step(d, average_speed, cumulated_pos, food, enemy, N, 0,
                         d->local_n, base_random);
  } else {
    
#ifdef USE_OPENMP
//printf("%d threads\n", NR_THREADS);
// Compute in parallel if openmp
#pragma omp parallel num_threads(NR_THREADS)
    {
      
      unsigned int rank = omp_get_thread_num();
      //printf("rank=%d\n", rank);
      unsigned int base = ratio * rank;
      unsigned int limit = ratio * (rank + 1);
      inner_dragonfly_step(d, average_speed, cumulated_pos, food, enemy, N,
                           base, limit, base_random);
    }
#else
    // Compute it in serial
    for (unsigned int rank = 0; rank < NR_THREADS; rank++) {
      unsigned int base = ratio * rank;
      unsigned int limit = ratio * (rank + 1);
      inner_dragonfly_step(d, average_speed, cumulated_pos, food, enemy, N,
                           base, limit, base_random);
    }
#endif

    if (rest != 0) {
      unsigned int r_base = ratio * NR_THREADS;
      unsigned int r_end = d->local_n;

      inner_dragonfly_step(d, average_speed, cumulated_pos, food, enemy, N,
                           r_base, r_end, base_random);
    }
  }
  weights_step(&d->w);
}*/
/*
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
      printf("k=%d\n", k);
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
      printf("kend=%d\n", k);
    }
  }
  printf("END FIRST CYCLE\n");
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
  printf("END Function\n");
  fflush(stdout);
}

void dragonfly_compute_step(Dragonfly *d, float *average_speed,
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
}*/

void inner_computation_accumulate(Dragonfly* d,float* cumulated_pos,float* cumulated_speed,float* next_food_fitness,
																	float* next_enemy_fitness,float* best_fitness,
																	unsigned int* indexes,unsigned int base,
																	unsigned int limit,unsigned int dim,unsigned int seed){
		
		for (unsigned int k = base; k < limit; k++) {
			float* iter_pos = d->positions + dim * k;
			float* iter_speed = d->speeds + dim * k;
			sum_assign(cumulated_pos,iter_pos,dim);
			sum_assign(cumulated_speed,iter_speed,dim);
						
			float fitness = d->fitness(iter_pos, &seed, dim);

			if (fitness > *next_food_fitness) {
				indexes[0] = k;
				*next_food_fitness = fitness;
			}
			if (fitness < *next_enemy_fitness) {
				indexes[1] = k;
				*next_enemy_fitness = fitness;
			}
			if (fitness > *best_fitness) {
				indexes[2] = k;
				*best_fitness = fitness;
			}
		}

}

void update_status(ComputationStatus* status,Dragonfly* d,float* best,float* best_fitness,unsigned int* indexes,unsigned int dim){
		memcpy(status->next_food, d->positions + dim*indexes[0], sizeof(float) * dim);
		memcpy(status->next_enemy, d->positions + dim*indexes[1], sizeof(float) * dim);
		memcpy(best, d->positions + dim*indexes[2], sizeof(float) * dim);

		status->next_food_fitness = d->fitness(status->next_food,&d->seed,dim);
		status->next_enemy_fitness = d->fitness(status->next_enemy,&d->seed,dim);
		*best_fitness = d->fitness(best,&d->seed,dim);
}
/*
void computation_accumulate(ComputationStatus *status, Dragonfly *d,
                            float *best, float *best_fitness,
                            unsigned int NUM_THREADS) {

  unsigned int dim = d->dim;
	zeroed(status->cumulated_pos, dim);
  zeroed(status->cumulated_speeds, dim);
  memcpy(status->next_enemy, d->positions, sizeof(float) * dim);
  memcpy(status->next_food, d->positions, sizeof(float) * dim);
  status->next_enemy_fitness = d->fitness(status->next_enemy, &d->seed, dim);
  status->next_food_fitness = status->next_enemy_fitness;
  status->n = d->local_n;

	unsigned int pos_indexes[3] = {0,0,0};

#ifdef USE_OPENMP
	unsigned int rest = d-> % NUM_THREADS;
  unsigned int ratio = d->starting_chunk_count / NUM_THREADS;
	unsigned int g_seed = rand_r(&d->seed);

#pragma omp parallel num_threads(NUM_THREADS)
	{
		
		unsigned int rank = omp_get_thread_num();
    unsigned int base = ratio * rank;
    unsigned int limit = ratio * (rank + 1);
		unsigned int local_seed = g_seed+rank;

		float local_cumulated_pos[dim];
		float local_cumulated_speeds[dim];


		unsigned int local_indexes[3] = {base,base,base};

		float next_enemy_fitness;
		float next_food_fitness;
		float local_best_fitness;
		float *cur_pos = d->positions + dim*base;
		
		next_enemy_fitness = d->fitness(cur_pos,&local_seed,dim);

		next_food_fitness = next_enemy_fitness;
		local_best_fitness = next_enemy_fitness;

		zeroed(local_cumulated_pos,dim);
		zeroed(local_cumulated_speeds,dim);
		
		inner_computation_accumulate(d,local_cumulated_pos,local_cumulated_speeds,&next_food_fitness,
																	&next_enemy_fitness,&local_best_fitness,
																	local_indexes,base,
																	limit,dim,local_seed);


#pragma omp critical
		{
			sum_assign(status->cumulated_pos,local_cumulated_pos,dim);
			sum_assign(status->cumulated_speeds,local_cumulated_speeds,dim);

			if (next_food_fitness > status->next_food_fitness) {
				pos_indexes[0] = local_indexes[0];
				status->next_food_fitness = next_food_fitness;
			}
			if (next_enemy_fitness < status->next_enemy_fitness) {
				pos_indexes[1] = local_indexes[1];
				status->next_enemy_fitness = next_enemy_fitness;
			}
			if (local_best_fitness > *best_fitness) {
				pos_indexes[2] = local_indexes[2];
				*best_fitness = local_best_fitness;
			}
		}

	}

	update_status(status,d,best,best_fitness,pos_indexes,dim);
	

	if(rest!=0){

    unsigned int r_base = ratio * NUM_THREADS;
		unsigned int local_seed = g_seed + r_base;

		inner_computation_accumulate(d,status->cumulated_pos,status->cumulated_speeds,&status->next_food_fitness,
																	&status->next_enemy_fitness,best_fitness,
																	pos_indexes,r_base,
																	d->N,dim,local_seed);


		update_status(status,d,best,best_fitness,pos_indexes,dim);
	}
#else
  (void)(NUM_THREADS);
  
	inner_computation_accumulate(d,status->cumulated_pos,status->cumulated_speeds,&status->next_food_fitness,
																	&status->next_enemy_fitness,best_fitness,
																	pos_indexes,0,d->local_n,dim,d->seed);

	update_status(status,d,best,best_fitness,pos_indexes,dim);

#endif
}*/

/*
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
            "Cannot call computation accumulate serial with more than 1
threads\n"); exit(1);

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
}*/
