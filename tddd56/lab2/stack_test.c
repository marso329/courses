/*
 * stack_test.c
 *
 *  Created on: 18 Oct 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 * 
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 * 
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stddef.h>

#include "stack.h"
#include "non_blocking.h"

#define test_run(test)\
  printf("[%s:%s:%i] Running test '%s'... ", __FILE__, __FUNCTION__, __LINE__, #test);\
  test_setup();\
  if(test())\
  {\
    printf("passed\n");\
  }\
  else\
  {\
    printf("failed\n");\
  }\
  test_teardown();

typedef int data_t;
#define DATA_SIZE sizeof(data_t)
#define DATA_VALUE 5

stack_t *stack;
data_t data;

#if MEASURE != 0
struct stack_measure_arg
{
  int id;
};
typedef struct stack_measure_arg stack_measure_arg_t;

struct timespec t_start[NB_THREADS], t_stop[NB_THREADS], start, stop;

#if MEASURE == 1
void*
stack_measure_pop(void* arg)
{
  stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
  int i;
  node_t** nodes=malloc(sizeof(node_t*)* MAX_PUSH_POP / NB_THREADS);
  clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
  for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
  {
    // See how fast your implementation can pop MAX_PUSH_POP elements in parallel
    nodes[i]=stack_pop(stack);
  }
  clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);
  for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
  {
    // See how fast your implementation can pop MAX_PUSH_POP elements in parallel
    if (nodes[i]!=NULL){
    free(nodes[i]);
    }
  }
  free(nodes);

  return NULL;
}
#elif MEASURE == 2
void*
stack_measure_push(void* arg)
{
  stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
  int i;
  node_t* nodes[MAX_PUSH_POP / NB_THREADS];
  for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
  {
    nodes[i]=malloc(sizeof(node_t));
    nodes[i]->data=i;
  }
  clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
  for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
  {
    // See how fast your implementation can push MAX_PUSH_POP elements in parallel
    stack_push(stack,nodes[i]);
  }
  clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);

  return NULL;
}
#endif
#endif

/* A bunch of optional (but useful if implemented) unit tests for your stack */
void test_init() {
  // Initialize your test batch
//stack = malloc(sizeof(stack_t));
  // stack_init(stack);
}

void test_setup() {
  stack = malloc(sizeof(stack_t));
  stack_init(stack);

}

void test_teardown() {
  // Do not forget to free your stacks after each test
  // to avoid memory leaks
  stack_destroy(stack);
  free(stack);
}

void test_finalize() {
  stack_destroy(stack);
}

int test_push_safe() {
  // Make sure your stack remains in a good state with expected content when
  // several threads push concurrently to it

  // Do some work
  node_t* data = malloc(sizeof(node_t));
  data->data = 0;

  stack_push(stack, data);

  // check if the stack is in a consistent state
  stack_check(stack);

  // check other properties expected after a push operation
  // (this is to be updated as your stack design progresses)
  // assert(stack->change_this_member == 0);

  // For now, this test always fails
  data = stack_pop(stack);
  int number = data->data;
  free(data);
  return number == 0;
}

struct pop_push_test_args {
stack_t* stack;
int* counter;
pthread_mutex_t *lock;
};
typedef struct pop_push_test_args pop_push_test_args_t;

void test_thread_pop(void* arg){
	pop_push_test_args_t *args = (pop_push_test_args_t*) arg;
	int i;
	node_t* data;
#if NON_BLOCKING == 0

	args->stack->stack_lock=args->lock;
#endif
	for (i=0;i<=10000;i++){
		printf("poping\n");
		data=stack_pop(args->stack);
		printf("done poping \n");
		if(data!=NULL){
			if (data->data!=1000){
			printf("error detected\n");
			}
		*(args->counter)+=1;
		free(data);
		}
	}
}
void test_thread_push(void* arg){
	pop_push_test_args_t *args = (pop_push_test_args_t*) arg;
	node_t* data;
#if NON_BLOCKING == 0

	args->stack->stack_lock=args->lock;
#endif
	int i;
	for (i=0;i<=1000;i++){
		printf("pushing\n");
		data =malloc(sizeof(node_t));
		data->data=1000;
		stack_push(args->stack,data);
		printf("done pushing\n");
	}

}

int test_push_pop_safe(){
	  pthread_t thread[2];

	  pthread_attr_t attr;
	  pthread_mutexattr_t mutex_attr;
	  pthread_attr_init(&attr);
	  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	  pthread_mutexattr_init(&mutex_attr);

	  pthread_mutex_t lock;
	  pthread_mutex_init(&lock, NULL);


	  pop_push_test_args_t* pop=malloc(sizeof(pop_push_test_args_t));
	  pop_push_test_args_t* push=malloc(sizeof(pop_push_test_args_t));
	  stack_t* own_stack = malloc(sizeof(stack_t));
	  stack_init(own_stack);
	  pop->stack=own_stack;
	  push->stack=own_stack;
	  pop->lock=&lock;
	  push->lock=&lock;
	  pop->counter=malloc(sizeof(int));
	  *(pop->counter)=0;
	  pthread_create(&thread[1],  &attr, &test_thread_push, (void*) &push);
	  pthread_create(&thread[0], &attr, &test_thread_pop, (void*) &pop);
	  pthread_join(thread[0], NULL);
	  pthread_join(thread[1], NULL);
	  printf("the number of successfulls pop are:%i \n",pop->counter);

return 1;
}


int test_pop_safe() {
	  // Do some work
	  node_t* data = malloc(sizeof(node_t));
	  data->data = 0;

	  stack_push(stack, data);

	  // check if the stack is in a consistent state
	  stack_check(stack);

	  // check other properties expected after a push operation
	  // (this is to be updated as your stack design progresses)
	  // assert(stack->change_this_member == 0);

	  // For now, this test always fails
	  data = stack_pop(stack);
	  int number = data->data;
	  free(data);
	  return number == 0;
}

int own_simple_test() {
  stack_t* own_stack = malloc(sizeof(stack_t));
  stack_init(own_stack);
  size_t i;
  node_t* data;

  for (i = 1; i <= 10; i++) {
    data = malloc(sizeof(node_t));
    data->data = i;
    stack_push(own_stack, data);
  }
  int pass = 1;
  for (i = 10; i > 0; i--) {
    data = stack_pop(own_stack);
    if (data->data != i) {
      pass = 0;
    }
    free(data);
  }
  stack_destroy(own_stack);
  free(own_stack);
  return pass;
}

// 3 Threads should be enough to raise and detect the ABA problem
#define ABA_NB_THREADS 2

// We test here the CAS function
struct thread_test_aba_args {
  int id;
  size_t* counter;
  pthread_mutex_t *lock;
  stack_t* shared_stack;
  int success;
};
typedef struct thread_test_aba_args thread_test_aba_args_t;

void*
thread_test_aba_1(void* arg) {
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  thread_test_aba_args_t *args = (thread_test_aba_args_t*) arg;
  pthread_mutex_lock(args->lock);
  printf("thread 1 is in control and begins poping 10 from stack\n");
  node_t* data;
  while (1) {
    data=*(args->shared_stack->head);
    if (data==NULL){
      break;
    }
    int value=data->data;
    pthread_mutex_unlock(args->lock);
    sleep(1);
    pthread_mutex_lock(args->lock);
    if (cas(args->shared_stack->head,data,data->next)==data) {
      printf("thread one successfully pop:ed %i from stack\n",data->data);
      pthread_mutex_unlock(args->lock);
      if (value!=data->data){
      printf("aba problem detected\n");
      args->success=1;
      }

      break;
    }
  }
args->id=0;
#endif
  return NULL;
}

void*
thread_test_aba_2(void* arg) {
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  thread_test_aba_args_t *args = (thread_test_aba_args_t*) arg;
  pthread_mutex_lock(args->lock);
  printf("thread two is in controll\n");
  node_t* data;
  while (1) {
    data=*(args->shared_stack->head);
    if (data==NULL){
      break;
    }
    if (cas(args->shared_stack->head,data,data->next)==data) {
      printf("thread two successfully poped %i from stack\n",data->data);
      break;
    }
  }
  node_t* node=malloc(sizeof(node_t));
  node->data=15;
  while (1) {
    node_t* temp=*(args->shared_stack->head);
    node->next=temp;
    if(cas(args->shared_stack->head,temp,node)==temp) {
      printf("thread 2 succesfully pushed 15 to stack successfull\n");
      break;
    }
  }
  printf("thread 2 modified data in %i" ,data->data);
  data->data=20;
  printf("to 20 \n");

  while (1) {
    node_t* temp=*(args->shared_stack->head);
    data->next=temp;
    if(cas(args->shared_stack->head,temp,data)==temp) {
      printf("thread 2 pushed 20 to stack \n");
      break;
    }
  }
  printf("thread two gives control back to thread one \n");
  pthread_mutex_unlock(args->lock);


#endif
  return NULL;
}


int test_aba() {
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  int success, aba_detected = 1;


  pthread_t thread[ABA_NB_THREADS];
  thread_test_aba_args_t args[ABA_NB_THREADS];
  pthread_mutex_t lock;
  pthread_mutex_init(&lock, NULL);
  stack_t* shared_stack=malloc(sizeof(stack_t));
  stack_init(shared_stack);
  pthread_mutex_lock(&lock);


int i;
node_t* data;
for (i=0;i<=10;i++){
 data=malloc(sizeof(node_t));
 data->data=i;
 stack_push(shared_stack,data);
}
printf("\n");
printf("inital stack contains 10,9,8,7,6,5,4,3,2,1,0\n");

  for (i = 0; i < ABA_NB_THREADS; i++)
  {
    args[i].id = i;
    args[i].success=0;
    args[i].shared_stack=shared_stack;
    args[i].counter = malloc(sizeof(int));
    *(args[i].counter)=0;
    args[i].lock = &lock;
    if (i%2==1){
    pthread_create(&thread[i], NULL, &thread_test_aba_1, (void*) &args[i]);
    }
    else{
      pthread_create(&thread[i], NULL, &thread_test_aba_2, (void*) &args[i]);
    }
  }
  pthread_mutex_unlock(&lock);
  for (i = 0; i < ABA_NB_THREADS; i++)
  {
    pthread_join(thread[i], NULL);
  }
  node_t* printer;
  while(1){
    printer=stack_pop(shared_stack);
    if (printer==NULL){
    break;
    }
    printf("%i ",printer->data);
  }
  printf("\n");
  for (i = 0; i < ABA_NB_THREADS; i++)
  {
	    if(args[i].success==1){
	    	success=0;
	    }

  }



  return success;
#else
  // No ABA is possible with lock-based synchronization. Let the test succeed only
  return 1;
#endif
}

// We test here the CAS function
struct thread_test_cas_args {
  int id;
  size_t* counter;
  pthread_mutex_t *lock;
};
typedef struct thread_test_cas_args thread_test_cas_args_t;

void*
thread_test_cas(void* arg) {
#if NON_BLOCKING != 0
  thread_test_cas_args_t *args = (thread_test_cas_args_t*) arg;
  int i;
  size_t old, local;

  for (i = 0; i < MAX_PUSH_POP; i++)
  {
    do {
      old = *args->counter;
      local = old + 1;
#if NON_BLOCKING == 1
    }while (cas(args->counter, old, local) != old);
#elif NON_BLOCKING == 2
  }while (software_cas(args->counter, old, local, args->lock) != old);
#endif
}
#endif

  return NULL;
}

// Make sure Compare-and-swap works as expected
int test_cas() {
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  thread_test_cas_args_t args[NB_THREADS];
  pthread_mutexattr_t mutex_attr;
  pthread_mutex_t lock;

  size_t counter;

  int i, success;

  counter = 0;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutex_init(&lock, &mutex_attr);

  for (i = 0; i < NB_THREADS; i++)
  {
    args[i].id = i;
    args[i].counter = &counter;
    args[i].lock = &lock;
    pthread_create(&thread[i], &attr, &thread_test_cas, (void*) &args[i]);
  }

  for (i = 0; i < NB_THREADS; i++)
  {
    pthread_join(thread[i], NULL);
  }

  success = counter == (size_t)(NB_THREADS * MAX_PUSH_POP);

  if (!success)
  {
    printf("Got %ti, expected %i. ", counter, NB_THREADS * MAX_PUSH_POP);
  }

  assert(success);
  return success;
#else
  return 1;
#endif
}

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
// MEASURE == 0 -> run unit tests
#if MEASURE == 0
  test_init();

  test_run(test_cas);

  test_run(test_push_safe);
  test_run(test_pop_safe);
  test_run(test_aba);
  test_run(own_simple_test)
  //test_run(test_push_pop_safe);

  test_finalize();
#else
  int i;
  pthread_t thread[NB_THREADS];
  pthread_attr_t attr;
  stack_measure_arg_t arg[NB_THREADS];

  test_setup();
  pthread_attr_init(&attr);

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (i = 0; i < NB_THREADS; i++)
  {
    arg[i].id = i;
#if MEASURE == 1
    pthread_create(&thread[i], &attr, stack_measure_pop, (void*)&arg[i]);
#else
    pthread_create(&thread[i], &attr, stack_measure_push, (void*)&arg[i]);
#endif
  }

  for (i = 0; i < NB_THREADS; i++)
  {
    pthread_join(thread[i], NULL);
  }
  clock_gettime(CLOCK_MONOTONIC, &stop);

  // Print out results
  for (i = 0; i < NB_THREADS; i++)
  {
    printf("%i %i %li %i %li %i %li %i %li\n", i, (int) start.tv_sec,
        start.tv_nsec, (int) stop.tv_sec, stop.tv_nsec,
        (int) t_start[i].tv_sec, t_start[i].tv_nsec, (int) t_stop[i].tv_sec,
        t_stop[i].tv_nsec);
  }
#endif

  return 0;
}
