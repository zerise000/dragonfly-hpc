CC=gcc
CFLAGS=-Wall -Wextra -g -o -fsanitize=address
EXEC=dragonfly-serial
DLL=-lm

serial: main.c utils.c
	@$(CC) $(CFLAGS) -o $@ $^ $(DLL) 

parallel: main.c utils.c parallel-utils.c
	@$(CC) $(CFLAGS) -o $@ $^ $(DLL) 

.PHONY clean: serial
	@rm $^ 
