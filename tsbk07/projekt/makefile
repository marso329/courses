LDIR=lib
SDIR=src
INCLUDE_DIR = includes
BUILD_DIR = obj
CC=g++
MAKEFLAGS += --jobs=4
CFLAGS= -std=c++11 -Wall -O3
LIBS= -lGL  -lGLEW -lX11 -lboost_system -lboost_filesystem  -lopencv_core -lopencv_imgproc -L/usr/local/lib/  -lopencv_imgcodecs 
SRCS = $(wildcard $(SDIR)/*.cpp)
OBJS = $(patsubst %.cpp, %.o, $(subst $(SDIR)/, $(BUILD_DIR)/, $(SRCS)))
LRCS = $(wildcard $(LDIR)/*.c)
OBJS += $(patsubst %.c, %.o, $(subst $(LDIR)/, $(BUILD_DIR)/, $(LRCS)))


all: main
.PHONY : all

main: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS)  -o main  $(LIBS) 

$(BUILD_DIR)/%.o: $(SDIR)/%.cpp $(wildcard $(INCLUDE_DIR)/*.h)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I $(INCLUDE_DIR) -I $(LDIR) -c $< -o $@  

$(BUILD_DIR)/%.o: $(LDIR)/%.c $(wildcard $(LDIR)/*.h)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I $(LDIR) -I $(INCLUDE_DIR) -c $< -o $@ 

.PHONY: clean
clean :
	rm -f main *.o *~>//dev/null
	rm -rf $(BUILD_DIR)
