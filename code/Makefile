CC=gcc
CFLAGS=-Wall -Wextra -g -o -fsanitize=address -std=c11 
DLL=-lm 

serial.bin: main.c utils.c
	@$(CC) $(CFLAGS) -o $@ $^ $(DLL) 

openmpi.bin: openmpi.c utils.c parallel-utils.c
	@$(CC) $(CFLAGS) -o $@ $^ $(DLL) -fopenmp 

.PHONY clean: serial
	@rm $^ 
