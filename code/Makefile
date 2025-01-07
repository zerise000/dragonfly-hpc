CC=gcc
CFLAGS=-Wall -Wextra -fopenmp -g -o -std=c11 
EXEC=dragonfly-parallel
DLL=-lm
SRC=main.c utils.c parallel.c wrappers.h

$(EXEC): $(SRC) 
	@$(CC) $(CFLAGS) -o $@ $^ $(DLL) 

.PHONY clean: $(EXEC) 
	@rm $^ 
