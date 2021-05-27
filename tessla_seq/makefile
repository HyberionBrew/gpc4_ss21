CC = g++
CFLAGS = -g --std=c++14
LDFLAGS =
SRC_DIR = src/
TEST_DIR = test/
TARGET_DIR = bin/
OBJS = StreamFunctions.o Debug.o main.o Stream.o Writer.o

all: main
	rm -f *.o

main: $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET_DIR)$@ $^ $(LDFLAGS)

%.o: $(SRC_DIR)%.cpp $(SRC_DIR)%.h
	$(CC) $(CFLAGS) $< -c -o $@

test: main
	$(CC) $(CFLAGS) -o $(TEST_DIR)run_test $(OBJS) $(TEST_SRC)
	@$(TEST_DIR)run_test

clean:
	rm -f bin/*
	rm -f *.o
