CC = g++
CFLAGS = -g --std=c++14
LDFLAGS =
SRC_DIR = src/
TARGET_DIR = bin/

all: main
	rm -f *.o

main: StreamFunctions.o Debug.o main.o Writer.o
	$(CC) $(CFLAGS) -o $(TARGET_DIR)$@ $^ $(LDFLAGS)

%.o: $(SRC_DIR)%.cpp $(SRC_DIR)%.h
	$(CC) $(CFLAGS) $< -c -o $@

clean:
	rm -f bin/*
	rm -f *.o
