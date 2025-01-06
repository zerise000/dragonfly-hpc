CC=gcc
CFLAGS=-Wall -Wextra -fopenmp -g -o -std=c11 
EXEC=dragonfly-parallel
DLL=-lm

$(EXEC): main.c utils.c parallel.c
	@$(CC) $(CFLAGS) -o $@ $^ $(DLL) 

.PHONY clean: $(EXEC) 
	@rm $^ 
