#include <stdlib.h>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <string.h>
#include <stack>
#include <mutex>
#include <thread>
#include <ctime> 
#include <chrono>  
using namespace std;
#define NB_THREADS 4

//Struct to give information about jobs to working threads;
struct job_t{
  int start;
  int end;
  int* data;
  job_t(int start, int end,int* data) : data(data), start(start),end(end) {}
  int depth;
};

//main function of quicksort
void quicksort(stack<job_t*>* job_stack,mutex* stack_lock,int* numbers_done,int* numbers_to_do);


//stuff
void sort(int* array, size_t size);

static
void simple_quicksort(int *array, size_t size);
