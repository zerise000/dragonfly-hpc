CC=gcc
CFLAGS=-Wall -Wextra -g -o -fsanitize=address
EXEC=dragonfly-serial
DLL=-lm

$(EXEC): main.c utils.c
	@$(CC) $(CFLAGS) -o $@ $^ $(DLL) 

.PHONY clean: $(EXEC) 
	@rm $^ 
