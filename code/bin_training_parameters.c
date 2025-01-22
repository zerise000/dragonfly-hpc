#include "dragonfly-common.h"
#include "utils.h"
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DA_SERIAL_LIB
#include "bin_serial.c"

float eval(float *wi){
    Weights w = {
      // exploring
      .al = {wi[0], wi[1]},
      .cl = {wi[2], wi[3]},
      // swarming
      .sl = {wi[4], wi[5]},
      .fl = {wi[6], wi[7]},
      .el = {wi[8], wi[9]},
      .wl = {wi[10], wi[11]},
      .ll = {wi[12], wi[13]},
      .max_speedl = {wi[14], wi[15]},
  };
  Parameters p ={.n=100, .dim=10, .chunks=8, .iterations=1000};
  

  Fitness fitness = shifted_fitness;
  float avg=0.0;
  int N = 10;
  for(int i =0; i<N; i++){
    unsigned int seed = rand();

    float *shifted_tmp = malloc(sizeof(float)*p.dim);
    float *shifted_rotation = malloc(sizeof(float)*p.dim*p.dim);
    float *shifted_shift = init_array(p.dim, 100.0, &seed);
    init_matrix(shifted_rotation, 100.0, p.dim, &seed);
    
    init_shifted_fitness(shifted_tmp, shifted_rotation, shifted_shift, rastrigin_fitness);
    
    float *res = dragonfly_serial_compute(p, w, fitness, seed);
    //printf("%f\n", fitness(res, p.dim));
    avg+= fitness(res, p.dim);
    free(shifted_tmp);
    free(shifted_rotation);
    free(shifted_shift);
    free(res);
  }
 //printf("avg: %f\n", avg);
    return avg/N;
}

int main(){
    /*
    Weights w = {
      // exploring
      .al = {0.3, 0.00},
      .cl = {0.00, 0.3},
      // swarming
      .sl = {0.4, 0.0},
      .fl = {0.7, 0.7},
      .el = {0.0, 0.0},
      .wl = {0.9, 0.2},
      .ll = {0.2, 0.3},
      .max_speedl = {2.0, 2.0},
  };
    */
    float best[16] = {0.3, 0.0, 0.0, 0.3, 0.4, 0.0, 0.7, 0.7, 0.0, 0.0, 0.9, 0.2, 0.2, 0.3, 2.0, 2.0};
    float best_fitness = eval(best);
    float cur[16];
    unsigned int seed = rand();
    while(true){
        for(int i=0; i<16; i++){
            cur[i]=best[i];
            //if(rand()%8==0){
                cur[i]=best[i]+RAND_FLOAT(0.05, &seed);
                if(cur[i]<0.0){
                    cur[i]=0.0;
                }
            //}
            
        }
        float fit = eval(cur);
        if(fit>best_fitness){
            best_fitness=fit;
            printf("New min %f\n", best_fitness);
            for(int i=0; i<16; i++){
                printf("%f ", cur[i]);
            }
            printf("\n");
            memcpy(best, cur, sizeof(float)*16);
        }
    }

}
/*
New min -353743.656250
0.181880 -0.102300 -0.035865 0.361603 0.596336 0.018159 0.651550 0.487240 -0.064944 0.030771 0.898261 0.121631 0.063970 0.295120 1.880680 2.105216

New min -1655767.000000
0.343304 0.120054 0.147655 0.456489 0.523672 0.009136 0.579197 0.649009 0.000000 0.011433 0.626118 0.226548 0.070795 0.611347 2.347073 1.677076 
*/
