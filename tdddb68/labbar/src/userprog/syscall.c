#include "userprog/syscall.h"
#include <stdio.h>
#include <syscall-nr.h>
#include "threads/interrupt.h"
#include "threads/thread.h"
#include "threads/synch.h"
#include "threads/init.h"
#include "threads/malloc.h"
#include "filesys/filesys.h"
#include "filesys/file.h"
#include "lib/syscall-nr.h"
#include "process.h"
#include "devices/input.h"
#include "threads/vaddr.h"
#include "userprog/pagedir.h"

static void syscall_handler(struct intr_frame *);

void halt(void);
bool create(const char *filename, unsigned file_size);
int open(const char *file);
void close(int fd);
int read(int fd, void *buffer, unsigned size);
int write(int fd, const void *buffer, unsigned size);
void exit(int status);
void close_all_files(void);
int exec(const char *cmd_line);
int wait(int pid);
struct child_parent* get_child_process(int pid);
void remove_child_process(struct child_parent *cp);
void remove_dead_children(void);
void check_argument(void* esp);
void seek(int fd, int position);
int tell(int fd);
int filesize(int fd);
bool remove(const char *file);
void check_string (const void* string);
void check_buffer(const void* buffer,int size);
struct process_file {
	struct file *file;
	int fd;
	struct list_elem elem;
};

void syscall_init(void) {
	intr_register_int(0x30, 3, INTR_ON, syscall_handler, "syscall");
}

static void
syscall_handler (struct intr_frame *f UNUSED)
{
	void *sp = f->esp;
	check_argument(sp);
	int syscall_no = *(int*) sp;
	int fd;
	const char *filename;
	switch(syscall_no) {

		case SYS_HALT:
		halt();
		break;

		case SYS_CREATE:
		check_argument(sp+4);
		check_argument(*(const char**) (sp+4));
		check_argument(sp+8);

		check_string(*(const void **) (sp+4));

		filename = *(const char**) (sp+4);
		unsigned initial_size = *(unsigned*) (sp+8);
		f->eax = create(filename, initial_size);
		break;

		case SYS_REMOVE:
		check_argument(sp+4);
		check_argument(*(const char**) (sp+4));
		check_string(*(const void**) (sp+4));
		filename = *(const char**) (sp+4);
		f->eax=remove(filename);
		break;

		case SYS_OPEN:
		check_argument(sp+4);
		check_argument(*(const char**) (sp+4));
		check_string(*(const void**) (sp+4));

		filename = *(const char**) (sp+4);
		f->eax = open(filename);
		break;

		case SYS_CLOSE:
		fd = *(int*) (sp+4);
		close(fd);
		break;

		case SYS_READ:
		check_argument(sp+4);
		check_argument(sp+8);
		check_argument(sp+12);
		fd = *(int*) (sp+4);
		void *buf = *(void**) (sp+8);
		check_argument(buf);
		unsigned r_size = *(unsigned*) (sp+12);
		check_buffer(buf,r_size);
		f->eax = read(fd, buf, r_size);
		break;

		case SYS_SEEK:
		check_argument(sp+4);
		check_argument(sp+8);
		fd = *(int*) (sp+4);
		int position = *(int*) (sp+8);
		seek(fd, position);
		break;

		case SYS_TELL:
		check_argument(sp+4);
		fd = *(int*) (sp+4);
		f->eax=tell(fd);
		break;

		case SYS_FILESIZE:
		check_argument(sp+4);
		fd = *(int*) (sp+4);
		f->eax=filesize(fd);
		break;

		case SYS_WRITE:
		check_argument(sp+4);
		check_argument(sp+8);
		check_argument(sp+12);
		fd = *(int*) (sp+4);
		const void *buffer = *(const void**) (sp+8);
		check_argument(buffer);
		unsigned w_size = *(unsigned*) (sp+12);
		check_buffer(buffer,w_size);
		f->eax = write(fd, buffer, w_size);
		break;

		case SYS_EXIT:
		check_argument(sp+4);
		int status = *(int*) (sp+4);
		exit(status);
		break;

		case SYS_EXEC:
		  check_argument(sp+4);
		  check_argument(*(const char**) (sp+4));
		  check_string(*(const void**) (sp+4));
		  filename = *(const char**) (sp+4);
		  f->eax = exec(filename);
		break;

		case SYS_WAIT:
		f->eax = wait(*(int*) (sp+4));
		break;

		default:
		printf("System call undefined.\n");
	}
}

int tell(int fd) {
	if (bitmap_test(thread_current()->file_bitmap, fd - 2) == 1) {
		struct file* file = thread_current()->file_array[fd];
		return file_tell(file);
	}
	return -1;
}


bool remove(const char *file) {
	return filesys_remove(file);
}

int filesize(int fd) {
	if (bitmap_test(thread_current()->file_bitmap, fd - 2) == 1) {
		struct file* file = thread_current()->file_array[fd];
		return file_length(file);
	}
	return -1;
}

void seek(int fd, int position) {
	if (bitmap_test(thread_current()->file_bitmap, fd - 2) == 1) {
		struct file* file = thread_current()->file_array[fd];
		file_seek(file, position);
	}
}

void check_argument(void* esp) {
	if (esp == NULL || !is_user_vaddr(esp)
			|| pagedir_get_page(thread_current()->pagedir, esp) == NULL) {
		exit(-1);
	}
}

int wait(int pid) {
	return process_wait(pid);
}


void check_string (const void* string)
{
  while (* (char *) string != NULL)
    {
	  (char *) string = (char *) string + 1;
      check_argument(string);
    }
}
void check_buffer(const void* buffer,int size){
	int i =0;
	  for (;i<size;i++ )
	    {
		  (char *) buffer = (char *) buffer + 1;
	      check_argument(buffer);
	    }

}

struct child_parent* get_child_process(int pid) {
	struct thread *t = thread_current();
	struct list_elem* e;
	struct child_parent* cp;
	for (e = list_begin(t->children); e != list_end(t->children);
			e = list_next(e)) {
		cp = list_entry (e, struct child_parent, elem);
		if (pid == cp->pid) {
			return cp;
		}
	}
	return NULL;
}

void remove_dead_children(void) {
	struct thread *t = thread_current();
	struct list_elem *e;
	struct child_parent *cp;
	for (e = list_begin(t->children); e != list_end(t->children);
			e = list_next(e)) {
		cp = list_entry (e, struct child_parent, elem);
		sema_down(&cp->ref_cntr);
		if (cp->reference_counter < 2) {
			remove_child_process(cp);
		} else {
			cp->reference_counter--;
			sema_up(&cp->ref_cntr);
		};
	}
}

void remove_child_process(struct child_parent *cp) {
	list_remove(&cp->elem);
	free(cp);
}

void halt(void) {
	close_all_files();
	power_off();
}

int exec(const char *cmd_line) {
	return process_execute(cmd_line);
}

/* Creates a new file called file initially initial_size bytes in size.
 * Returns true if successful, false otherwise. */
bool create(const char *name, unsigned initial_size) {
	return filesys_create(name, initial_size);
}

/* Opens the file called file. 
 * Returns a nonnegative integer handle called a "file descriptor" (fd), or -1 if the file could not be opened. */
int open(const char *file) {
	if (thread_current()->fd > 126) {
		return -1;
	}
	struct file *f = filesys_open(file);
	if (f == NULL) {
		return -1;
	}
	thread_current()->fd++;
	int fd = bitmap_scan_and_flip(thread_current()->file_bitmap, 0, 1, 0) + 2;
	thread_current()->file_array[fd] = f;
	return fd;
}

/* Closes file descriptor fd. */
void close(int fd) {
	if (fd < 2 || fd > 126) {
		exit(-1);
	}
	if (bitmap_test(thread_current()->file_bitmap, fd - 2) == 1) {

		thread_current()->fd--;
		file_close((thread_current()->file_array[fd]));
		bitmap_set(thread_current()->file_bitmap, fd - 2, false);
	} else {
		exit(-1);
	}
}

/* Reads size bytes from the file open as fd into buffer.
 * Returns the number of bytes actually read, or -1 if the file could not be read. */
int read(int fd, void *buffer, unsigned size) {
	if (size < 1) {
		return 0;
	}
	if (fd == STDOUT_FILENO || fd > 126) {
		exit(-1);
	}
	if (fd == STDIN_FILENO) {
		// input_getc() returns uint8_t, which is the same as a byte
		unsigned int i;
		for (i = 0; i < size; ++i) {
			*(char*) (buffer + i) = (char) input_getc();
		}
		return size;
	} else {
		if (bitmap_test(thread_current()->file_bitmap, fd - 2) == 1) {
			struct file* file = thread_current()->file_array[fd];
			int res = file_read(file, buffer, size);
			if (res == 0) {
				return -1;
			}
			return res;
		}
		//printf("Read FEL!");
		return -1;
	}
}

/* Writes size bytes from buffer to the open file fd.
 * Returns the actual number written or -1 if no bytes could be written at all. */
int write(int fd, const void *buffer, unsigned size) {
	if (size < 1) {
		return 0;
	}
	if (fd == STDIN_FILENO || fd > 126) {
		exit(-1);
	} else if (fd == STDOUT_FILENO) {
		putbuf((char *) buffer, size);	
		return size;
	} else {
		if (bitmap_test(thread_current()->file_bitmap, fd - 2) == 1) {
			struct file* file = thread_current()->file_array[fd];
			int res = file_write(file, buffer, size);
			if (res == 0) {
				return -1;
			}
			return res;
		}
		return -1;
	}
}

/* Terminates the current user program, returning status to the kernel. */
void exit(int status) {
	thread_current()->parent->exit_code = status;
	//free our list of children
	printf("%s: exit(%d)\n", thread_current()->name, status);
	thread_exit();
}

/* Closes all open files */
void close_all_files(void) {
	if (thread_current()->file_bitmap != NULL) {
		while (BITMAP_ERROR
				!= bitmap_scan(thread_current()->file_bitmap, 0, 126, 1)) {
			close(
					bitmap_scan_and_flip(thread_current()->file_bitmap, 0, 1, 1)
							+ 2);
		}
		bitmap_destroy(thread_current()->file_bitmap);
	}
}
