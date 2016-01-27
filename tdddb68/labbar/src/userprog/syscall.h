#ifndef USERPROG_SYSCALL_H
#define USERPROG_SYSCALL_H
#include "threads/thread.h"
void syscall_init (void);
void remove_child_process (struct child_parent *cp);
void remove_dead_children (void);
struct child_parent* get_child_process (int pid);
void close_all_files (void);
void remove_dead_children(void);
void remove_child_process(struct child_parent *cp);
struct child_parent* get_child_process(int pid);

#endif /* userprog/syscall.h */
