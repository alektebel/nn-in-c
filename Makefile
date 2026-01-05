# Compiler to use
CC = gcc

# Common compiler flags: warnings, debug info, optimization
CFLAGS = -Wall -g -O2

# Default target: build the executable
main: main.c
	$(CC) $(CFLAGS) -o main main.c -lm

# Optional: clean up the built executable
clean:
	rm -f main

# Prevent make from confusing these targets with files
.PHONY: clean
