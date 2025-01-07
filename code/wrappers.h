#ifndef __WRAPPERS_H__
#define __WRAPPERS_H__

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

typedef struct{
	float* pos;
	float* next;
	float next_fitness;
} Entity;

typedef struct{
	float* cumulated_pos;
	float* average_speed;
} Temp;

typedef struct {
  float sl[2], al[2], cl[2], fl[2], el[2], wl[2];
  float st, at, ct, ft, et, wt;
  float s, a, c, f, e, w;
} Weights;

typedef struct {
	float* positions;
	float* speeds;
	unsigned int dim;
}Dragonfly;

#endif
