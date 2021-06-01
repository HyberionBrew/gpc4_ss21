CC = g++
CFLAGS = -g --std=c++14 -pthread
LDFLAGS =
SRC_DIR = src/
TEST_DIR = test/
TARGET_DIR = bin/
OBJS = main.o Decode.o InstrInterface.o Scheduler.o DebugScheduler.o GPUScheduler.o lfqueue.o

all: main
	rm -f *.o
	mv bin/main bin/arc

main: $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET_DIR)$@ $^ $(LDFLAGS)

%.o: $(SRC_DIR)%.cpp $(SRC_DIR)%.h
	$(CC) $(CFLAGS) $< -c -o $@

lfqueue.o: $(SRC_DIR)lfqueue.c $(SRC_DIR)lfqueue.h
	gcc $(CFLAGS) $< -c -o $@

test: main
	$(CC) $(CFLAGS) -o $(TEST_DIR)run_test $(OBJS) $(TEST_SRC)
	@$(TEST_DIR)run_test

clean:
	rm -f bin/*
	rm -f *.o
