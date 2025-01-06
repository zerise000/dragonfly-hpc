CC=gcc
CFLAGS=-Wall -Wextra -g -o -fsanitize=address -std=c11 
DLL=-lm 

serial: main.c utils.c
	@$(CC) $(CFLAGS) -o $@ $^ $(DLL) 

parallel: main.c utils.c parallel-utils.c
	@$(CC) $(CFLAGS) -o $@ $^ $(DLL) -fopenmp 

.PHONY clean: serial
	@rm $^ 
