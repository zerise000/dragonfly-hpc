CC=gcc
CFLAGS=-Wall -Wextra -g -o -fsanitize=address -std=c11 
DLL=-lm 

serial.bin: main.c utils.c dragonfly-common.c
	@$(CC) $(CFLAGS) -o $@ $^ $(DLL) 

openmpi.bin: openmpi.c utils.c parallel-utils.c
	@$(CC) $(CFLAGS) -o $@ $^ $(DLL) -fopenmp 

.PHONY clean: 
	@rm serial.bin openmpi.bin
