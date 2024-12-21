#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>

#define POP_SIZE 1000
#define RAND_FLOAT(N) ((float)rand()/(float)RAND_MAX)*N
#define ZEROS(N) init_array(N,0);

// take timing not including IO

typedef struct{
  float x;
  float y;
} Coord;


Coord* init_array(int dim,int range_max){
	Coord *arr = (Coord*)malloc(dim*sizeof(Coord));

	for(int i=0; i<POP_SIZE; i++){
      arr[i] = (Coord){.x = RAND_FLOAT(range_max), .y = RAND_FLOAT(range_max)}; 
	}

	return arr;
}


float distance(Coord p1,Coord p2){
  double dx = p1.x - p2.x;
  double dy = p1.y - p2.y;

  return sqrt(pow(dy,2)+pow(dx,2)); 
}


bool get_updates(Coord* position,Coord* step,Coord* food,Coord* enemy_pos,float* weights,Coord* updates){
  bool neighbours = false;
  Coord S;
  Coord A;
  Coord C;
  Coord F;
  Coord E;
  int nr_neighbours;


  float eps = 10e-2;

  for(int dragonfly=0; dragonfly<POP_SIZE; dragonfly++){
    S = (Coord){.x = 0,.y=0};
    A = (Coord){.x = 0,.y=0};
    C = (Coord){.x = 0,.y=0};
    F = (Coord){.x = 0,.y=0};
    E = (Coord){.x = 0,.y=0};
    nr_neighbours = 0;

    for(int k=0; k<POP_SIZE; k++){

      if(k != dragonfly && distance(position[dragonfly],position[k]) < eps){
            S.x += position[k].x - position[dragonfly].x;
            S.y += position[k].y - position[dragonfly].y;

            A.x += step[k].x;
            A.y += step[k].y;

            C.x += position[k].x;
            C.y += position[k].y;

            nr_neighbours++;
        }

    }

    F.x = food[dragonfly].x - position[dragonfly].x;
    F.y = food[dragonfly].y - position[dragonfly].y;

    E.x = enemy_pos[dragonfly].x - position[dragonfly].x;
    E.y = enemy_pos[dragonfly].y - position[dragonfly].y;

    updates[dragonfly].x = (weights[0]*S.x + weights[3]*F.x + weights[4]*E.x) + weights[5]*step[dragonfly].x;

    updates[dragonfly].y = (weights[0]*S.y + weights[3]*F.y + weights[4]*E.y) + weights[5]*step[dragonfly].y;


    if(nr_neighbours > 0){
        A.x /= nr_neighbours;
        A.y /= nr_neighbours;

        C.x /= nr_neighbours;
        C.y /= nr_neighbours;

        C.x -= position[dragonfly].x;
        C.y -= position[dragonfly].y;

        neighbours = true;

        updates[dragonfly].x += weights[1]*A.x + weights[2]*C.x;
        updates[dragonfly].y +=  weights[1]*A.y +weights[2]*C.y;
    }
  }


  return neighbours;
}

float sigma(float beta) {
  return tgamma(beta+1)*sin(beta*M_PI/2)/tgamma((beta+1)/2)*beta*pow(2,(beta-1)/2);
}

float levy(int dim){
  float beta = 4.25;
  return 0.01*(RAND_FLOAT(1)*sigma(beta)/pow(RAND_FLOAT(1),1/beta));
}

void update_with_formula(Coord* position,Coord* step,Coord* updates){
  for(int dragonfly = 0; dragonfly<POP_SIZE; dragonfly++){
    step[dragonfly].x = updates[dragonfly].x;
    step[dragonfly].y = updates[dragonfly].y;

    position[dragonfly].x += step[dragonfly].x;
    position[dragonfly].y += step[dragonfly].y;
  }
}


void update_random(Coord* position){
  for(int dragonfly = 0; dragonfly < POP_SIZE; dragonfly++){
     position[dragonfly].x += levy(POP_SIZE)*position[dragonfly].x;
     position[dragonfly].y += levy(POP_SIZE)*position[dragonfly].y;
  }
}

int main() {
   
	srand(time(NULL));

	Coord* position = init_array(POP_SIZE,POP_SIZE);
	Coord* step = ZEROS(POP_SIZE)
	Coord* enemy_pos = init_array(POP_SIZE,POP_SIZE);
    Coord* food = init_array(POP_SIZE,POP_SIZE); 
    Coord* updates = ZEROS(POP_SIZE)
    
    
    float weights[6] = {
      RAND_FLOAT(1),
      RAND_FLOAT(1),
      RAND_FLOAT(1),
      RAND_FLOAT(1),
      RAND_FLOAT(1),
      RAND_FLOAT(1)
    };

	int iter_max = 100;
    bool use_levy = false;
	
	for(int i=0; i<iter_max; i++){
      use_levy = get_updates(position,step,food,enemy_pos,weights,updates);


      if(!use_levy)
        update_with_formula(position,step,updates);
      else
        update_random(position);
             
	}


    free(updates);
    free(food);
	free(step);
	free(position);
	free(enemy_pos);

	return 0;
}
