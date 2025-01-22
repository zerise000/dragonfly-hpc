CC=clang
CFLAGS=-Wall -Wextra -o -fsanitize=address -std=c11  -D_POSIX_C_SOURCE=200112L -O3
#-O3
DLL=-lm 

all: serial.bin openmpi.bin mpich.bin

serial.bin: bin_serial.c utils.c dragonfly-common.c
	@$(CC) $(CFLAGS) -o $@ $^ $(DLL) 

training.bin: bin_training_parameters.c utils.c dragonfly-common.c
	@$(CC) $(CFLAGS) -o $@ $^ $(DLL) 

openmpi.bin: bin_openmpi.c utils.c parallel-utils.c
	@$(CC) $(CFLAGS) -o $@ $^ $(DLL) -fopenmp

mpich.bin: bin_mpich.c utils.c dragonfly-common.c
	mpicc $(CFLAGS) -o $@ $^ $(DLL) -fopenmp

.PHONY clean: 
	@rm serial.bin openmpi.bin mpich.bin
