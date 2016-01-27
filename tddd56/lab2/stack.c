/*
 * stack.c
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
 *     but WITHOUT ANY WARRANTY without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "stack.h"
#include "non_blocking.h"

#if NON_BLOCKING == 0
#warning Stacks are synchronized through locks
#else
#if NON_BLOCKING == 1
#warning Stacks are synchronized through hardware CAS
#else
#warning Stacks are synchronized through lock-based CAS
#endif
#endif

void stack_check(stack_t *stack) {
// Do not perform any sanity check if performance is bein measured
#if MEASURE == 0
  // Use assert() to check if your stack is in a state that makes sens
  // This test should always pass
  assert(1 == 1);

  // This test fails if the task is not allocated or if the allocation failed
  assert(stack != NULL);
#endif
}
void stack_destroy(stack_t* stack){
#if NON_BLOCKING == 0
  while (stack->head!=NULL){
    stack_pop(stack);
  }
  pthread_mutex_destroy(stack->stack_lock);
#endif
#if NON_BLOCKING == 1
  if(stack->head==NULL){
    return;
  }
  while (*(stack->head)!=NULL){
    stack_pop(stack);
  }
  //free(stack->head);
#endif
  // Destroy properly your test batch
}

void stack_init(stack_t* stack) {
  // Allocate a new stack and reset its values
#if NON_BLOCKING == 0
  stack->head = NULL;
  stack->stack_lock=malloc(sizeof(pthread_mutex_t));
  pthread_mutex_init(stack->stack_lock, NULL);
#endif
#if NON_BLOCKING == 1
  node_t** node_pointer = malloc(sizeof(node_t*));
  *node_pointer=NULL;
  stack->head=node_pointer;
#endif

}

int /* Return the type you prefer */
stack_push(stack_t* stack, node_t* node) {
  if (stack==NULL){
    return -1;
  }
#if NON_BLOCKING == 0
  pthread_mutex_lock(stack->stack_lock);
  node->next = stack->head;
  stack->head = node;
  pthread_mutex_unlock(stack->stack_lock);
#elif NON_BLOCKING == 1
  while (1) {
    node_t* temp=*(stack->head);
    node->next=temp;
    if(cas(stack->head,temp,node)==temp) {
      break;
    }
  }
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses
  stack_check((stack_t*) 1);

  return 0;
}

node_t* /* Return the type you prefer */
stack_pop(stack_t* stack) {
#if NON_BLOCKING == 0
  pthread_mutex_lock(stack->stack_lock);
  if (stack->head == NULL) {
    pthread_mutex_unlock(stack->stack_lock);
    return NULL;
  }
  node_t* old_node = stack->head;
  stack->head = old_node->next;
  pthread_mutex_unlock(stack->stack_lock);
  return old_node;

#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
  node_t* data;
  while (1) {
    data=*(stack->head);
    if (data==NULL){
      return NULL;
    }
    if (cas(stack->head,data,data->next)==data) {
      break;
    }
  }
  return data;
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  return 0;
}

