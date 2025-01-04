#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include "utils.h"

// take timing not including IO
struct Weights{
  float sl[2], al[2], cl[2], fl[2], el[2], wl[2];
  float st, at, ct, ft, et, wt;
  float s, a, c, f, e, w;
};
typedef struct Weights Weights;

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
float* dragonfly(unsigned int dimensions, unsigned int N, unsigned int iterations, float space_size, Weights weights, float (*fitness)(float*, unsigned int)){
  //compute weigths progression
  compute_steps(&weights, iterations);
  
  //allocate, and init random positions
  float* positions = init_array(N*dimensions, space_size);
  float* speeds = init_array(N*dimensions, space_size/20.0);

  // allocate food and next_food, 
  float* food = init_array(dimensions, space_size);
  float* next_food = malloc(dimensions*sizeof(float));
  memcpy(next_food, food, dimensions*sizeof(float));
  float next_food_fitness = fitness(food, dimensions);

  // allocate enemy and next_enemy 
  float* enemy = init_array(dimensions, space_size);
  float* next_enemy = init_array(dimensions, space_size);
  memcpy(next_enemy, enemy, dimensions*sizeof(float));
  float next_enemy_fitness=fitness(enemy, dimensions);

  //some temp values.
  float* cumulated_pos = init_array(dimensions, 0.0);
  float* tmp = init_array(dimensions, 0.0);
  float* avarage_speed = init_array(dimensions, 0.0);

  float* S = init_array(dimensions, 0.0);
  float* A = init_array(dimensions, 0.0);
  float* C = init_array(dimensions, 0.0);
  float* F = init_array(dimensions, 0.0);
  float* E = init_array(dimensions, 0.0);
  float* W = init_array(dimensions, 0.0);
  float* delta_pos = init_array(dimensions, 0.0);

  //for each iteration
  for(unsigned int i=0; i<iterations; i++){
    //compute avarage speed and positions
    zeroed(cumulated_pos, dimensions);
    zeroed(avarage_speed, dimensions);
    for(unsigned int j=0; j<N;j++){
      sum_assign(cumulated_pos, positions+dimensions*j, dimensions);
      sum_assign(avarage_speed, speeds+dimensions*j, dimensions);
    }
    scalar_prod_assign(avarage_speed, 1.0/(float)N, dimensions);

    //for each dragonfly
    for(unsigned int j=0; j<N;j++){
      float* cur_pos = positions+dimensions*j;
      float* cur_speed = speeds+dimensions*j;
      //compute separation: Si = -sumall(X-Xi)
      memcpy(S, cur_pos, sizeof(float)*dimensions);
      scalar_prod_assign(S, -(float)N, dimensions);
      sum_assign(S, cumulated_pos, dimensions);
      scalar_prod_assign(S, weights.s, dimensions);

      //compute alignament: Ai = avarage(Vi)
      memcpy(A, avarage_speed, sizeof(float)*dimensions);
      scalar_prod_assign(A, weights.a, dimensions);

      //compute cohesion: Ci = avarage(Xi)-X
      memcpy(C, cumulated_pos, sizeof(float)*dimensions);
      scalar_prod_assign(C, 1.0/(float)N, dimensions);
      sub_assign(C, cur_pos, dimensions);
      scalar_prod_assign(C, weights.c, dimensions);

      //food attraction: Fi=X_food - X
      memcpy(F, food, sizeof(float)*dimensions);
      sub_assign(F, cur_pos, dimensions);
      scalar_prod_assign(F, weights.f, dimensions);

      //enemy repulsion: E=X_enemy+X
      memcpy(E, enemy, sizeof(float)*dimensions);
      sum_assign(E, cur_pos, dimensions);
      scalar_prod_assign(E, weights.e, dimensions);

      //compute speed = sSi + aAi + cCi + fFi + eEi + w
      scalar_prod_assign(cur_speed, weights.w, dimensions);
      sum_assign(cur_speed, E, dimensions);
      sum_assign(cur_speed, F, dimensions);
      sum_assign(cur_speed, C, dimensions);
      sum_assign(cur_speed, A, dimensions);
      sum_assign(cur_speed, S, dimensions);

      //check if speed is too big
      if (length(cur_speed, dimensions)>space_size/10.0){
        float speed = length(cur_speed, dimensions);
        scalar_prod_assign(cur_speed, space_size/10.0/speed, dimensions);
      }


      //update current pos
      sum_assign(cur_pos, cur_speed, dimensions);
      float fit = fitness(cur_pos, dimensions);
      //printf("%f\n", fit);
      if(fit<next_enemy_fitness){
        next_enemy_fitness=fit;
        memcpy(next_enemy, cur_pos, dimensions*sizeof(float));
      }
      if(fit>next_food_fitness){
        next_food_fitness=fit;
        memcpy(next_food, cur_pos, dimensions*sizeof(float));
      }
    }

    // update food and enemy
    //printf("found fitness=%f\n", next_food_fitness);
    memcpy(enemy, next_enemy, dimensions*sizeof(float));
    memcpy(food, next_food, dimensions*sizeof(float));

    //update weights
    step(&weights);
  }



  free(positions);
  free(speeds);
	free(food);
  //free(next_food);
	free(enemy);
  free(next_enemy);

  free(cumulated_pos);
  free(avarage_speed);
  free(tmp);

  free(S);
  free(A);
  free(C);
  free(F);
  free(E);
  free(W);
  free(delta_pos);
  return next_food;
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
  unsigned int dim=2;
  float* res = dragonfly(dim, 10000, 10000, 2.0, w, sphere_fitness);
  float fit = sphere_fitness(res, dim);
  printf("found fitness=%f\n", fit);
  for(unsigned int i=0; i<dim; i++){
    printf("%f\n", res[i]);
  }
  free(res);


	
}