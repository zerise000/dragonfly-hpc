#include "parallel.h"
#include "utils.h"
#include "wrappers.h"


void compute_steps(Weights* w, unsigned int steps){
  w->st=(w->sl[1]-w->sl[0])/(float)steps;
  w->s=w->sl[0];
  w->at=(w->al[1]-w->al[0])/(float)steps;
  w->a=w->al[0];

  w->ct=(w->cl[1]-w->cl[0])/(float)steps;
  w->c=w->cl[0];
  w->ft=(w->fl[1]-w->fl[0])/(float)steps;
  w->f=w->fl[0];

  w->et=(w->el[1]-w->el[0])/(float)steps;
  w->e=w->el[0];
  w->wt=(w->wl[1]-w->wl[0])/(float)steps;
  w->w=w->wl[0];
}

void step(Weights* w){
  w->s+=w->st;
  w->a+=w->at;
  w->c+=w->ct;
  w->f+=w->ft;
  w->e+=w->et;
  w->w+=w->wt;
}

Entity init_entity(unsigned int dimensions,float space_size,int nr_threads,float (*fitness)(float*, unsigned int)){
	Entity tmp;
	tmp.pos = init_array_parallel(dimensions, space_size,nr_threads);
  	tmp.next = malloc(dimensions*sizeof(float));
  	memcpy(tmp.next, tmp.pos, dimensions*sizeof(float));
  	tmp.next_fitness = fitness(tmp.pos, dimensions);
	return tmp;
}


float* dragonfly(unsigned int dimensions, unsigned int N, unsigned int iterations,unsigned int nr_threads, float space_size, Weights weights, float (*fitness)(float*, unsigned int)){
  //compute weigths progression
  compute_steps(&weights, iterations);
  
  //allocate, and init random positions
  Dragonfly dragonflies;
  dragonflies.positions = init_array_parallel(N*dimensions, space_size,nr_threads);
  dragonflies.speeds = init_array_parallel(N*dimensions, space_size/20.0, nr_threads);
  dragonflies.dim = N;

  // allocate food 
  Entity food = init_entity(dimensions,space_size,nr_threads,fitness);

  // allocate enemies
  Entity enemy= init_entity(dimensions,space_size,nr_threads,fitness);

  //some temp values.
  Temp temp;
  temp.cumulated_pos = init_array_parallel(dimensions, 0.0 ,nr_threads);
  temp.average_speed = init_array_parallel(dimensions, 0.0 ,nr_threads);

  //for each iteration
  for(unsigned int i=0; i<iterations; i++){
    //compute avarage speed and positions
	parallel_sum(&temp,dragonflies,N,dimensions,nr_threads);

    //for each dragonfly update food and enemy
	update(&food,&enemy,temp,dragonflies,weights,space_size,dimensions,nr_threads,fitness);

    //printf("found fitness=%f\n", next_food_fitness);
    memcpy(enemy.pos, enemy.next, dimensions*sizeof(float));
    memcpy(food.pos, food.next, dimensions*sizeof(float));

    //update weights
    step(&weights);
  }

  free(dragonflies.positions);
  free(dragonflies.speeds);

  free(enemy.pos);
  free(enemy.next);
  free(food.pos);

  free(temp.cumulated_pos);
  free(temp.average_speed);

 
  return food.next;
} 

float sphere_fitness(float* inp, unsigned int dim){
    float res=0.0;
    for(unsigned int i=0; i<dim; i++){
      res+=inp[i]*inp[i];
    }
    return -res;
}



int main() {
	srand(time(NULL));

	unsigned int dim = 2;	
	unsigned int tot_dragonflies = 10000; 
	unsigned int iters = 10000; 
	unsigned int nr_threads = 7; 

	Weights w ={
		//exploring
		.al={0.3, 0.01},
		.cl={0.01, 0.3},
		//swarming
		.sl={0.1, 0.1},
		.fl={0.1, 0.1},
		.el={0.1, 0.1},
		.wl={0.1, 0.1},
	};

	float* res = dragonfly(dim, tot_dragonflies, iters, nr_threads, 2.0, w, sphere_fitness);
	float fit = sphere_fitness(res, dim);
	printf("found fitness=%f\n", fit);
	for(unsigned int i=0; i<dim; i++){
		printf("%f\n", res[i]);
	}
	free(res);
	return 0;
}
