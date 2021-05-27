CC = g++
CFLAGS = -g --std=c++14
LDFLAGS =
SRC_DIR = src/
TEST_DIR = test/
TARGET_DIR = bin/
OBJS = Debug.o\
	Event.o\
	Reader.o\
	Stream.o\
	StreamFunctions.o\
	Writer.o

all: main
	rm -f *.o

main: main.o $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET_DIR)$@ $^ $(LDFLAGS)

%.o: $(SRC_DIR)%.cpp $(SRC_DIR)%.h
	$(CC) $(CFLAGS) $< -c -o $@

test: $(OBJS:.cpp=.o) $(TEST_DIR)test_main.o $(TEST_SRC:.cpp=.o) $(TEST_SRC)
	$(CC) $(CFLAGS) -o $(TEST_DIR)run_test $(OBJS) $(TEST_DIR)test_main.o $(TEST_SRC)
	@$(TEST_DIR)run_test

clean:
	rm -f bin/*
	rm -f *.o
	rm -f test/*.o
	rm -f test/run_test