CC=gcc
CFLAGS=-Wall -Wextra -g -o
EXEC=dragonfly-serial
DLL=-lm

$(EXEC): main.c 
	@$(CC) $(CFLAGS) $@ $^ $(DLL) 

.PHONY clean: $(EXEC) 
	@rm $^ 
