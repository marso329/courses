#include "filesys/free-map.h"
#include <bitmap.h>
#include <debug.h>
#include "filesys/file.h"
#include "filesys/filesys.h"
#include "filesys/inode.h"
#include "threads/synch.h"

static struct file *free_map_file;   /* Free map file. */
static struct bitmap *free_map;      /* Free map, one bit per disk sector. */
static struct semaphore modifier_map;

/* Initializes the free map. */
void
free_map_init (void) 
{
  free_map = bitmap_create (disk_size (filesys_disk));
  if (free_map == NULL)
    PANIC ("bitmap creation failed--disk is too large");
  bitmap_mark (free_map, FREE_MAP_SECTOR);
  bitmap_mark (free_map, ROOT_DIR_SECTOR);
  sema_init(&modifier_map,1);
}

/* Allocates CNT consecutive sectors from the free map and stores
   the first into *SECTORP.
   Returns true if successful, false if all sectors were
   available. */
bool
free_map_allocate (size_t cnt, disk_sector_t *sectorp) 
{
	sema_down(&modifier_map);
  disk_sector_t sector = bitmap_scan_and_flip (free_map, 0, cnt, false);
  if (sector != BITMAP_ERROR
      && free_map_file != NULL
      && !bitmap_write (free_map, free_map_file))
    {
      bitmap_set_multiple (free_map, sector, cnt, false); 
      sector = BITMAP_ERROR;
    }
  if (sector != BITMAP_ERROR){
    *sectorp = sector;
    }
  sema_up(&modifier_map);
  return sector != BITMAP_ERROR;
}

/* Makes CNT sectors starting at SECTOR available for use. */
void
free_map_release (disk_sector_t sector, size_t cnt)
{
	sema_down(&modifier_map);
  ASSERT (bitmap_all (free_map, sector, cnt));
  bitmap_set_multiple (free_map, sector, cnt, false);
  bitmap_write (free_map, free_map_file);
sema_up(&modifier_map);
}

/* Opens the free map file and reads it from disk. */
void
free_map_open (void) 
{
	sema_down(&modifier_map);
  free_map_file = file_open (inode_open (FREE_MAP_SECTOR));
  if (free_map_file == NULL)
    PANIC ("can't open free map");
  if (!bitmap_read (free_map, free_map_file))
    PANIC ("can't read free map");
  sema_up(&modifier_map);
}

/* Writes the free map to disk and closes the free map file. */
void
free_map_close (void) 
{
	sema_down(&modifier_map);
  file_close (free_map_file);
  sema_up(&modifier_map);
}

/* Creates a new free map file on disk and writes the free map to
   it. */
void
free_map_create (void) 
{
  /* Create inode. */
  if (!inode_create (FREE_MAP_SECTOR, bitmap_file_size (free_map)))
    PANIC ("free map creation failed");

  /* Write bitmap to file. */
  free_map_file = file_open (inode_open (FREE_MAP_SECTOR));
  if (free_map_file == NULL)
    PANIC ("can't open free map");
  if (!bitmap_write (free_map, free_map_file))
    PANIC ("can't write free map");
}