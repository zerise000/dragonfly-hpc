#include "parallel.h"
#include <omp.h>

float* init_array_parallel(unsigned int dimensions, float range_max,int nr_threads){

	float *arr = (float*)malloc(dimensions*sizeof(float));

#pragma omp parallel num_threads(nr_threads)
	{
		int ratio = dimensions/omp_get_num_threads();
		int rank = omp_get_thread_num(); 
		unsigned int base = ratio*rank;
		unsigned int limit = ratio*(rank+1);

#pragma omp parallel for
		for(unsigned int i=base; i<limit; i++){
		  arr[i] = RAND_FLOAT(range_max);
		}
	}
	return arr;

}


void update(Entity* food,Entity* enemy,Temp temp,Dragonfly dragonflies,Weights weights,float space_size,unsigned int dimensions,unsigned int nr_threads, float (*fitness)(float*, unsigned int)){
#pragma omp parallel num_threads(nr_threads)
{

	int ratio = dragonflies.dim/omp_get_num_threads();
	int rank = omp_get_thread_num(); 
	unsigned int base = ratio*rank;
	unsigned int limit = ratio*(rank+1);

	float cumulative_length;
	float S;
	float A;
	float C;
	float F;
	float E;

	float* cur_pos;
	float* cur_speed;

    for(unsigned int j=base; j<limit;j++){
	  cur_pos = dragonflies.positions+j*dimensions;
	  cur_speed = dragonflies.speeds+j*dimensions;

	  cumulative_length = 0;

      //compute speed = sSi + aAi + cCi + fFi + eEi + w
      scalar_prod_assign(cur_speed, weights.w, dimensions);

	  for(unsigned int i = 0; i<dimensions; i++){
		  S = ((-(float)dragonflies.dim)*cur_pos[i])+(temp.cumulated_pos[i]);
		  A = temp.average_speed[i];
		  C = temp.cumulated_pos[i]*(1/dragonflies.dim);
		  F = food->pos[i]-cur_pos[i];
		  E = enemy->pos[i]+cur_pos[i];

		  cur_speed[i] += weights.s*S;
		  cur_speed[i] += weights.a*A;
		  cur_speed[i] += weights.c*C;
		  cur_speed[i] += weights.f*F;
		  cur_speed[i] += weights.e*E;

		  cumulative_length += pow(cur_speed[i],2);
	  }

	  cumulative_length = sqrt(cumulative_length);

      //check if speed is too big
      if (cumulative_length>space_size/10.0){
        float speed = cumulative_length; 
        scalar_prod_assign(cur_speed, space_size/10.0/speed, dimensions);
      }


      //update current pos
      sum_assign(cur_pos, cur_speed, dimensions);
      float fit = fitness(cur_pos, dimensions);

      //printf("%f\n", fit);
      if(fit<enemy->next_fitness){
        enemy->next_fitness=fit;
        memcpy(enemy->next, cur_pos, dimensions*sizeof(float));
      }

      if(fit>food->next_fitness){
        food->next_fitness=fit;
        memcpy(food->next, cur_pos, dimensions*sizeof(float));
      }
    }
}

}

void parallel_sum(Temp* tmp_buffs,Dragonfly dragonflies,unsigned int N,unsigned int dimensions,unsigned int tot_threads){
    zeroed(tmp_buffs->cumulated_pos, dimensions);
    zeroed(tmp_buffs->average_speed, dimensions);

#pragma omp parallel num_threads(tot_threads)
	{
		float local_cumulated_pos[dimensions];
		float local_average_speed[dimensions];
		int rank = omp_get_thread_num();
		int ratio = N/tot_threads;

		int base = ratio*rank;
		unsigned int limit = ratio*(rank+1);

    	for(unsigned int j=base; j<limit; j++){
			float* curr_pos = dragonflies.positions+dimensions*j;
			float* curr_speed = dragonflies.speeds+dimensions*j;

			for(unsigned int k=0; k<dimensions; k++){
				local_cumulated_pos[k] += curr_pos[k];
				local_average_speed[k] += curr_speed[k];
			}
		}   
#pragma omp critical
		{
			for(unsigned int j=0; j<dimensions; j++){
				tmp_buffs->cumulated_pos[j] = local_cumulated_pos[j];
				tmp_buffs->average_speed[j] = local_average_speed[j]*(1/N);
			}
		}
 	}
}
