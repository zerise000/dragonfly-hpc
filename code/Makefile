CC=gcc
CFLAGS=-Wall -Wextra -std=c11  -D_POSIX_C_SOURCE=200112L -ggdb3
#-O3
DLL=-lm 

BINARIES=serial.bin training_mpich.bin mpich.bin training_parameters.bin

all: $(BINARIES)

serial.bin: bin_serial.c utils.c dragonfly-common.c
	$(CC) $(CFLAGS) -o $@ $^ $(DLL) 

training_mpich.bin: bin_training_mpich.c utils.c dragonfly-common.c
	mpicc $(CFLAGS) -DUSE_MPI -DUSE_OPENMP -o $@ $^ $(DLL) -fopenmp 

training_parameters.bin: bin_training_parameters.c utils.c dragonfly-common.c
	$(CC) $(CFLAGS) -o $@ $^ $(DLL)


mpich.bin: bin_mpich.c utils.c dragonfly-common.c
	mpicc $(CFLAGS) -DUSE_MPI -o $@ $^ $(DLL) -fopenmp

.PHONY clean: 
	@rm -f $(BINARIES)
