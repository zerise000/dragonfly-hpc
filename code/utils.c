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
#ifdef DUSE_OPENMP
#pragma omp parallel for reduction(+: dest)
#endif
  for (unsigned int i = 0; i < size; i++) {
    dest[i] += source[i];
  }
}

void sub_assign(float *dest, float *source, unsigned int size) {
#ifdef DUSE_OPENMP
#pragma omp parallel for reduction(-: dest)
#endif
  for (unsigned int i = 0; i < size; i++) {
    dest[i] -= source[i];
  }
}

void scalar_prod_assign(float *dest, float val, unsigned int size) {
#ifdef DUSE_OPENMP
#pragma omp parallel for reduction(*: dest)
#endif
  for (unsigned int i = 0; i < size; i++) {
    dest[i] *= val;
  }
}

float dot_product(float *a, float *b, unsigned int size) {
#ifdef DUSE_OPENMP
#pragma omp parallel for reduction(+: ret)
#endif
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

void inner_computation_accumulate(Dragonfly* d,unsigned int thread_start,unsigned int thread_end,
																	float* local_cumulated_pos,float* local_cumulated_speed,
																	float* local_food_fitness,float* local_enemy_fitness,
																	unsigned int* local_indexes,unsigned int dim,unsigned int *local_seed){
	for (unsigned int k = thread_start; k < thread_end; k++) {
		float *iter_pos = d->positions + dim * k;
		float *iter_speed = d->speeds + dim * k;
		sum_assign(local_cumulated_pos, iter_pos, dim);
		sum_assign(local_cumulated_speed, iter_speed, dim);

		float fitness = d->fitness(iter_pos,local_seed, dim);

		if (fitness > *local_food_fitness) {
			local_indexes[0] = k;
			*local_food_fitness = fitness;
		}
		if (fitness < *local_enemy_fitness) {
			local_indexes[1] = k;
			*local_enemy_fitness = fitness;
		}
	}
}

void update_status(Dragonfly* d,LogicalChunk* current_chunk,
									 float next_food_fitness,float next_enemy_fitness,
									 unsigned int* indexes,unsigned int start,
									 unsigned int end,unsigned int dim){
	memcpy(current_chunk->comp.next_food, d->positions + indexes[0] * dim,
         sizeof(float) * dim);
  memcpy(current_chunk->comp.next_enemy, d->positions + indexes[1] * dim,
         sizeof(float) * dim);
  current_chunk->comp.next_enemy_fitness = next_enemy_fitness;
  current_chunk->comp.next_food_fitness = next_food_fitness;
  current_chunk->comp.n = end - start;
}

// TODO paralelize
//  it computes the best, the food, the enemy, and the sums of speeds and
//  positions of an interval. the interval must be inside the current thread
//  chunk
void new_computation_accumulate(Dragonfly *d, LogicalChunk *current_chunk,
                                unsigned int *seed,unsigned int nr_threads) {
#ifdef USE_OPENMP
  unsigned int start = max(current_chunk->start, d->start);
  unsigned int end = min(current_chunk->end, d->end);
  unsigned int dim = d->dim;

  float next_enemy_fitness = d->fitness(d->positions, &d->seed, dim);
  float next_food_fitness = next_enemy_fitness;
  // status->n = d->local_n;

  end = end - d->start;
	start -= d->start;

  unsigned int indexes[2] = {0, 0};

  unsigned int thread_indexes[nr_threads][2]; 
	float thread_fitnesses[nr_threads][2];
	float thread_cumulated_pos[nr_threads][dim];
	float thread_cumulated_speed[nr_threads][dim];

  float *cumulated_pos = current_chunk->comp.cumulated_pos;
  float *cumulated_speed = current_chunk->comp.cumulated_speeds;
  zeroed(cumulated_pos, dim);
  zeroed(cumulated_speed, dim);
	
	unsigned int elems_per_thread = (end-start)/nr_threads;
	unsigned int rest = (end-start) % nr_threads;

#pragma omp parallel num_threads(nr_threads)
	{
		unsigned int thread_id = omp_get_thread_num();
		unsigned int thread_start = start+(thread_id*elems_per_thread);
		unsigned int thread_end = thread_start + elems_per_thread;

		thread_fitnesses[thread_id][0] = next_food_fitness;
		thread_fitnesses[thread_id][1] = next_enemy_fitness;
		unsigned int local_seed = (*seed) + thread_id;

		thread_indexes[thread_id][0] = thread_start;
		thread_indexes[thread_id][1] = thread_start;


		memcpy(thread_cumulated_pos[thread_id],cumulated_pos,dim*sizeof(float));
		memcpy(thread_cumulated_speed[thread_id],cumulated_speed,dim*sizeof(float));

		inner_computation_accumulate(d,thread_start,thread_end,
																	thread_cumulated_pos[thread_id],
																	thread_cumulated_speed[thread_id],
																	&thread_fitnesses[thread_id][0],
																	&thread_fitnesses[thread_id][1],
																	thread_indexes[thread_id],dim,&local_seed);
	}


	for(unsigned int i=0; i<nr_threads; i++){
		sum_assign(cumulated_pos,thread_cumulated_pos[i],dim);
		sum_assign(cumulated_speed,thread_cumulated_speed[i],dim);
		if(thread_fitnesses[i][0] > next_food_fitness){
			next_food_fitness = thread_fitnesses[i][0]; 
			indexes[0] = thread_fitnesses[i][0];
		}

		if(thread_fitnesses[i][1] < next_enemy_fitness){
			next_enemy_fitness = thread_fitnesses[i][1]; 
			indexes[1] = thread_fitnesses[i][1];
		}
	}

	if(rest>0){
		unsigned int rest_start = nr_threads*elems_per_thread+start;
		unsigned int rest_seed = (*seed)+rest_start;
	
		inner_computation_accumulate(d,rest_start,end,
																	cumulated_pos,cumulated_speed,
																	&next_food_fitness,&next_enemy_fitness,
																	indexes,dim,&rest_seed);
	
	}
	update_status(d,current_chunk,
									 next_food_fitness,next_enemy_fitness,
									 indexes,start,
									 end,dim);
 	
#else
  unsigned start = max(current_chunk->start, d->start);
  unsigned end = min(current_chunk->end, d->end);

  unsigned dim = d->dim;
  float *cumulated_pos = current_chunk->comp.cumulated_pos;
  float *cumulated_speed = current_chunk->comp.cumulated_speeds;

  zeroed(cumulated_pos, dim);
  zeroed(cumulated_speed, dim);

  float next_enemy_fitness = d->fitness(d->positions, &d->seed, dim);
  float next_food_fitness = next_enemy_fitness;
	start -= d->start;
  // status->n = d->local_n;

  unsigned int indexes[2] = {0, 0};
  end = end - d->start;
	inner_computation_accumulate(d,start,end,
																	cumulated_pos,cumulated_speed,
																	&next_food_fitness,&next_enemy_fitness,
																	indexes,dim,seed);
		
	update_status(d,current_chunk,
									 next_food_fitness,next_enemy_fitness,
									 indexes,start,
									 end,dim);
#endif
}


