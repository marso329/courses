semaphore mutex = 1;                 
semaphore db = 1;                    
int reader_count;                   

Reader()
{
  while (TRUE) {                    
     down(&mutex);                         
     reader_count = reader_count + 1;       
     if (reader_count == 1){
         down(&db);}                                                                    
     up(&mutex);                            
     read_db();                             
     down(&mutex);                          
     reader_count = reader_count - 1;       
     if (reader_count == 0){
         up(&db);}                                                    
     up(&mutex);}

Writer()
{
  while (TRUE) {               
     create_data();                        
     down(&db);                            
     write_db();                            
     up(&db);}
