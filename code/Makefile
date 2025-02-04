CC=gcc
CFLAGS=-Wall -g -Wextra -o -std=c11  -D_POSIX_C_SOURCE=200112L -O3
#-O3
DLL=-lm 

BINARIES=serial.bin training_mpich.bin mpich.bin

all: $(BINARIES)

serial.bin: bin_serial.c utils.c dragonfly-common.c utils-serial.c
	$(CC) $(CFLAGS) -o $@ $^ $(DLL) 

training_mpich.bin: bin_training_mpich.c utils.c dragonfly-common.c utils-serial.c
	mpicc $(CFLAGS) -DUSE_MPI -o $@ $^ $(DLL) -fopenmp 

training_parameters.bin: bin_training_parameters.c utils.c dragonfly-common.c utils-serial.c
	$(CC) $(CFLAGS) -o $@ $^ $(DLL)


mpich.bin: bin_mpich.c utils.c dragonfly-common.c utils-serial.c 
	mpicc $(CFLAGS) -DUSE_MPI -o $@ $^ $(DLL) 

.PHONY clean: 
	@rm -f $(BINARIES)
