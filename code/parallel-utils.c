#include "parallel-utils.h"
#include "utils.h"

#include <omp.h>
#include <stdlib.h>
#include <time.h>

float *init_array_parallel(unsigned int dimensions, float range_max,
                           int nr_threads) {

  float *arr = (float *)malloc(dimensions * sizeof(float));

#pragma omp parallel num_threads(nr_threads)
  {
    int ratio = dimensions / omp_get_num_threads();
    int rank = omp_get_thread_num();
    unsigned int base = ratio * rank;
    unsigned int limit = ratio * (rank + 1);

#pragma omp parallel for
    for (unsigned int i = base; i < limit; i++) {
      arr[i] = RAND_FLOAT(range_max);
    }
  }
  return arr;
}

void parallel_sum(float *cumulated_pos, float *average_speed, float *positions,
                  float *speeds, unsigned int N, unsigned int dimensions,
                  unsigned int tot_threads) {
  zeroed(cumulated_pos, dimensions);
  zeroed(average_speed, dimensions);
#pragma omp parallel num_threads(tot_threads)
  {
    float local_cumulated_pos[dimensions];
    float local_average_speed[dimensions];
    int rank = omp_get_thread_num();
    int ratio = N / tot_threads;

    int base = ratio * rank;
    unsigned int limit = ratio * (rank + 1);

    for (unsigned int j = base; j < limit; j++) {
      float *curr_pos = positions + dimensions * j;
      float *curr_speed = speeds + dimensions * j;

      for (unsigned int k = 0; k < dimensions; k++) {
        local_cumulated_pos[k] += curr_pos[k];
        local_average_speed[k] += curr_speed[k];
      }
    }
#pragma omp critical
    {
      for (unsigned int j = 0; j < dimensions; j++) {
        cumulated_pos[j] = local_cumulated_pos[j];
        average_speed[j] = local_average_speed[j] * (1 / N);
      }
    }
  }
}
