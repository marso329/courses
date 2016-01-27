#include <cstdio>
#include <algorithm>
#include <string.h>
#include "sort.h"
#include <stdlib.h>
#include <vector>
#include <stack>
#include <mutex>
#include <thread>
using namespace std;

//Struct to give information about jobs to working threads;
struct job_t {
  int start;
  int end;
  int* data;
  job_t(int start, int end, int* data) :
      data(data), start(start), end(end) {
  }
  int depth;
};

//main function of quicksort
void quicksort(stack<job_t*>* job_stack, mutex*, int* numbers_done,
    int* numbers_to_do, unique_lock<mutex>* uni_lock, condition_variable* cv);
static
void simple_quicksort(int *array, size_t size);

void quicksort(stack<job_t*>* job_stack, mutex* stack_lock, int* numbers_done,
    int* numbers_to_do, unique_lock<mutex>* uni_lock, condition_variable* cv) {
  printf("thread starting\n");
  job_t* my_job;
  bool quit = false;
  int temp_numbers_done = 0;
  int pivot = 0;
  int number_of_samples;
  int step;
  vector<int> pivots;
  int counter = 0;
  vector<int>* larger;
  vector<int>* smaller;
  vector<int>* equal;
  int left_start;
  int right_end;
  int left_end;
  int right_start;

  //do forever
  while (true) {
    temp_numbers_done = 0;
    pivot = 0;
    counter = 0;
    //get the job from the stack
    while (true) {
      //stack_lock->lock();

      if (*numbers_done >= *numbers_to_do) {
        quit = true;
        //stack_lock->unlock();
        break;
      }
      else if (!job_stack->empty()) {
        my_job = job_stack->top();
        job_stack->pop();
        //stack_lock->unlock();
        break;
      }
      else {
       // stack_lock->unlock();

      }

    }

    //if everything have been sorted we return
    if (quit) {
      printf("thread done\n");
      return;
    }

    //this should never happen
    if (my_job->start == my_job->end) {
      printf("thread done\n");
      return;
    }
    if (my_job->end - my_job->start < 10) {

      int temp_array[my_job->end - my_job->start];
      memcpy(temp_array, my_job->data + my_job->start,
          (my_job->end - my_job->start) * sizeof(int));
      simple_quicksort(temp_array, my_job->end - my_job->start);
      memcpy(my_job->data + my_job->start, temp_array,
          (my_job->end - my_job->start) * sizeof(int));

      stack_lock->lock();
      *numbers_done += my_job->end - my_job->start+1;
      stack_lock->unlock();
    }
    else {

      //sample and get a pivot
      number_of_samples = (my_job->end - my_job->start) / 100;
      if (number_of_samples > 100) {
        number_of_samples = 100;
      }
      if (number_of_samples <= 0) {
        pivot = my_job->data[my_job->start + (my_job->end - my_job->start) / 2];
      }
      else {
        step = (my_job->end - my_job->start) / number_of_samples;
        for (int i = my_job->start; i < my_job->end; i += step) {
          pivots.insert(pivots.begin(), my_job->data[i]);
        }
        sort(pivots.begin(), pivots.end());
        pivot = pivots[pivots.size() / 2];
      }
      vector<int> larger;
      vector<int> smaller;
      vector<int> equal;

      for (int i = my_job->start; i < my_job->end; i++) {
        if (my_job->data[i] > pivot) {
          larger.push_back(my_job->data[i]);
        }
        if (my_job->data[i] < pivot) {
          smaller.push_back(my_job->data[i]);
        }
        if (my_job->data[i] == pivot) {
          equal.push_back(my_job->data[i]);
        }
      }

      //transfer data to main array
      for (auto it = smaller.begin(); it != smaller.end(); it++) {
        my_job->data[my_job->start + counter] = *it;
        counter++;
      }
      for (auto it = equal.begin(); it != equal.end(); it++) {
        my_job->data[my_job->start + counter] = *it;
        counter++;
      }
      temp_numbers_done += equal.size() - 1;

      for (auto it = larger.begin(); it != larger.end(); it++) {
        my_job->data[my_job->start + counter] = *it;
        counter++;
      }

      //setup the limits for the two new jobs
      left_start = my_job->start;
      right_end = my_job->end;
      left_end = my_job->start + smaller.size();
      right_start = my_job->start + smaller.size() + equal.size();

      stack_lock->lock();
      if (left_start != left_end) {
        job_t* left_job = new job_t(left_start, left_end, my_job->data);
        job_stack->push(left_job);
      }
      else {
        temp_numbers_done += 1;
      }

      if (right_end != right_start) {
        job_t* right_job = new job_t(right_start, right_end, my_job->data);
        job_stack->push(right_job);
      }
      else {
        temp_numbers_done += 1;
      }
      *numbers_done += temp_numbers_done;
      if (*numbers_done >= *numbers_to_do) {
        quit = true;
      }

      stack_lock->unlock();
      delete my_job;
      if (quit) {
        break;
      }
    }
  }
  printf("thread done\n");
}

// These can be handy to debug your code through printf. Compile with CONFIG=DEBUG flags and spread debug(var)
// through your code to display values that may understand better why your code may not work. There are variants
// for strings (debug()), memory addresses (debug_addr()), integers (debug_int()) and buffer size (debug_size_t()).
// When you are done debugging, just clean your workspace (make clean) and compareile with CONFIG=RELEASE flags. When
// you demonstrate your lab, please cleanup all debug() statements you may use to faciliate the reading of your code.
#if defined DEBUG && DEBUG != 0
int *begin;
#define debug(var) printf("[%s:%s:%d] %s = \"%s\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#define debug_addr(var) printf("[%s:%s:%d] %s = \"%p\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#define debug_int(var) printf("[%s:%s:%d] %s = \"%d\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#define debug_size_t(var) printf("[%s:%s:%d] %s = \"%zu\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#else
#define show(first, last)
#define show_ptr(first, last)
#define debug(var)
#define debug_addr(var)
#define debug_int(var)
#define debug_size_t(var)
#endif

// A C++ container class that translate int pointer
// into iterators with little constant penalty
template<typename T>
class DynArray {
  typedef T& reference;
  typedef const T& const_reference;
  typedef T* iterator;
  typedef const T* const_iterator;
  typedef ptrdiff_t difference_type;
  typedef size_t size_type;

public:
  DynArray(T* buffer, size_t size) {
    this->buffer = buffer;
    this->size = size;
  }

  iterator begin() {
    return buffer;
  }

  iterator end() {
    return buffer + size;
  }

protected:
  T* buffer;
  size_t size;
};

static
void cxx_sort(int *array, size_t size) {
  DynArray<int> cppArray(array, size);
  std::sort(cppArray.begin(), cppArray.end());
}

// A very simple quicksort implementation
// * Recursion until array size is 1
// * Bad pivot picking
// * Not in place
static
void simple_quicksort(int *array, size_t size) {
  int pivot, pivot_count, i;
  int *left, *right;
  size_t left_size = 0, right_size = 0;

  pivot_count = 0;

  // This is a bad threshold. Better have a higher value
  // And use a non-recursive sort, such as insert sort
  // then tune the threshold value
  if (size > 1) {
    // Bad, bad way to pick a pivot
    // Better take a sample and pick
    // it median value.
    pivot = array[size / 2];

    left = (int*) malloc(size * sizeof(int));
    right = (int*) malloc(size * sizeof(int));

    // Split
    for (i = 0; i < size; i++) {
      if (array[i] < pivot) {
        left[left_size] = array[i];
        left_size++;
      }
      else if (array[i] > pivot) {
        right[right_size] = array[i];
        right_size++;
      }
      else {
        pivot_count++;
      }
    }

    // Recurse
    simple_quicksort(left, left_size);
    simple_quicksort(right, right_size);

    // Merge
    memcpy(array, left, left_size * sizeof(int));
    for (i = left_size; i < left_size + pivot_count; i++) {
      array[i] = pivot;
    }
    memcpy(array + left_size + pivot_count, right, right_size * sizeof(int));

    // Free
    free(left);
    free(right);
  }
  else {
    // Do nothing
  }
}

// This is used as sequential sort in the pipelined sort implementation with drake (see merge.c)
// to sort initial input data chunks before streaming merge operations.
void sort(int* array, size_t size) {

#if NB_THREADS == 0
  // Some sequential-specific sorting code
  simple_quicksort(array, size);
#else
  //if there is no data we return
  if (size == 0) {
    return;
  }

  //allocate memory
  stack<job_t*>* job_stack = new stack<job_t*>;
  mutex* stack_lock =new mutex;

  //setup condition variable
  mutex* cond_lock =new mutex;
  unique_lock<mutex>* lck=new unique_lock<mutex>(*cond_lock);

  condition_variable* cv=new condition_variable();
  //condition_variable cv;
  //do stuff
  job_t* main_job = new job_t(0, size, array);
  main_job->depth = 0;

  job_stack->push(main_job);
  int* number_done = (int*) malloc(sizeof(int));
  *number_done = 1;
  int* numbers_to_do = (int*) malloc(sizeof(int));
  *numbers_to_do = (int) size;
  //quicksort(job_stack, stack_lock, number_done, numbers_to_do);

  vector<thread> threads;
  for (int i =0;i<NB_THREADS;i++) {
    threads.push_back(thread(quicksort,job_stack,stack_lock,number_done,numbers_to_do,lck,cv));
  }
  //lck->unlock();
  for (int i =0;i<NB_THREADS;i++) {
    threads[i].join();
  }

  delete job_stack;
  delete stack_lock;
  delete cond_lock;
  delete cv;
  free(numbers_to_do);
  free(number_done);

#endif // #if NB_THREADS
}

