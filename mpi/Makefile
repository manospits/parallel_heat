MPICC = mpicc
FLAGS = -Wall
THREADS=4
OPENMP = -fopenmp -DTHREAD_COUNT=$(THREADS)
SIZE=900
BINS=heat_$(SIZE) heat_omp_$(SIZE) heat_con_$(SIZE) heat_con_omp_$(SIZE)
STEPS=10000
STEP=20
SOURCE = mpi_heat_improved_persistent_stat.c
all: $(BINS)

heat_$(SIZE): $(SOURCE)
	$(MPICC) -DNXPROB=$(SIZE) -DNYPROB=$(SIZE) -DSTEPS=$(STEPS) $(FLAGS) $? -o $@

heat_omp_$(SIZE): $(SOURCE)
	$(MPICC) -DNXPROB=$(SIZE) -DNYPROB=$(SIZE) -DSTEPS=$(STEPS) $(FLAGS) $(OPENMP) -DOMPCH $? -o $@

heat_con_$(SIZE): $(SOURCE)
	$(MPICC) -DNXPROB=$(SIZE) -DSTEP=$(STEP) -DNYPROB=$(SIZE) -DSTEPS=$(STEPS) $(FLAGS) -DCONVERGE $? -o $@

heat_con_omp_$(SIZE): $(SOURCE)
	$(MPICC) -DNXPROB=$(SIZE) -DSTEP=$(STEP) -DNYPROB=$(SIZE) -DSTEPS=$(STEPS) $(FLAGS) $(OPENMP) -DOMPCH -DCONVERGE $? -o $@

clean:
	rm -f $(BINS)
