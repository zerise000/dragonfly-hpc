#include "dragonfly-common.h"
#include "utils-special.h"
#include <omp.h>

void dragonfly_compute_step(Dragonfly *d, float *average_speed,
                            float *cumulated_pos, float *food, float *enemy,
                            unsigned int N){
unsigned int nr_threads = 2;
unsigned int rest = d->N % nr_threads;
unsigned int ratio = d->N / nr_threads;
unsigned int dimensions = d->dim;

#pragma omp parallel num_threads(nr_threads)
{

	unsigned int rank = omp_get_thread_num(); 
	unsigned int base = ratio*rank;
	unsigned int limit = ratio*(rank+1);

	float S;
	float A;
	float C;
	float F;
	float E;

	float* cur_pos;
	float* cur_speed;

    for(unsigned int j=base; j<limit;j++){
	  cur_pos = d->positions+j*dimensions;
	  cur_speed = d->speeds+j*dimensions;


      //compute speed = sSi + aAi + cCi + fFi + eEi + w
      scalar_prod_assign(cur_speed, d->w.w, dimensions);

	  for(unsigned int i = 0; i<dimensions; i++){
		  S = (-cur_pos[i])+(cumulated_pos[i]/((float)N));
		  A = average_speed[i];
		  C = cumulated_pos[i]*(1.0/(float)N);
		  F = food[i]-cur_pos[i];
		  E = enemy[i]+cur_pos[i];

		  cur_speed[i] += d->w.s*S;
		  cur_speed[i] += d->w.a*A;
		  cur_speed[i] += d->w.c*C;
		  cur_speed[i] += d->w.f*F;
		  cur_speed[i] += d->w.e*E;

	  }
    }
}
	if(rest != 0){
		unsigned int r_base = ratio*nr_threads;
		unsigned int r_end = d->N-1;

		for(unsigned int j = r_base; j<=r_end; j++){
		  float* cur_pos = d->positions+j*dimensions;
		  float* cur_speed = d->speeds+j*dimensions;

		  //compute speed = sSi + aAi + cCi + fFi + eEi + w
		  scalar_prod_assign(cur_speed, d->w.w, dimensions);

		  for(unsigned int i = 0; i<dimensions; i++){
			  float S = ((-(float)N)*cur_pos[i])+(cumulated_pos[i]);
			  float A = average_speed[i];
			  float C = cumulated_pos[i]*(1.0/(float)N);
			  float F = food[i]-cur_pos[i];
			  float E = enemy[i]+cur_pos[i];

			  cur_speed[i] += d->w.s*S;
			  cur_speed[i] += d->w.a*A;
			  cur_speed[i] += d->w.c*C;
			  cur_speed[i] += d->w.f*F;
			  cur_speed[i] += d->w.e*E;

		  }
		}
	}

  	weights_step(&d->w);

}

void computation_accumulate(ComputationStatus *message, Dragonfly *d, float* best, float* best_fitness){
  unsigned int dim=d->dim;
  unsigned int nr_threads = 2;
  unsigned int rest = d->N % nr_threads;
  unsigned int ratio = d->N / nr_threads;


  zeroed(message->cumulated_pos, dim);
  zeroed(message->cumulated_speeds, dim);
  memcpy(message->next_enemy, d->positions, sizeof(float) * dim);
  memcpy(message->next_food, d->positions, sizeof(float) * dim);
  message->next_enemy_fitness =
      d->fitness(message->next_enemy, dim);
  message->next_food_fitness = d->fitness(message->next_food, dim);
  message->n = 0;


#pragma omp parallel num_threads(nr_threads)
{
	int rank = omp_get_thread_num(); 
	unsigned int base = ratio*rank;
	unsigned int limit = ratio*(rank+1);

	for (unsigned int k = base; k < limit; k++) {
		float *cur_pos = d->positions + dim * k;
		sum_assign(message->cumulated_pos, cur_pos, dim);
		sum_assign(message->cumulated_speeds, d->speeds + dim * k, dim);

		float fitness = d->fitness(cur_pos, dim);
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

	if(rest != 0) {
		unsigned int r_base = ratio*nr_threads;
		unsigned int r_end = d->N-1;

		for (unsigned int k = r_base ; k <= r_end; k++) {
			float *cur_pos = d->positions + dim * k;
			sum_assign(message->cumulated_pos, cur_pos, dim);
			sum_assign(message->cumulated_speeds, d->speeds + dim * k, dim);
			float fitness = d->fitness(cur_pos, dim);
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
