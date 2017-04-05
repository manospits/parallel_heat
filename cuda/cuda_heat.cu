#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "timestamp.h"

#ifndef NXPROB
#define NXPROB 20
#endif
#ifndef NYPROB
#define NYPROB 20
#endif
#ifndef STEPS
#define STEPS 10000
#endif
#define CHECK_INTERVAL 20
#define THREADS_PER_ROW 32
#define THREADS_PER_BLOCK (THREADS_PER_ROW*THREADS_PER_ROW)
#define ROW_BLOCKS (((NXPROB-2)/THREADS_PER_ROW)+(((NXPROB-2)%THREADS_PER_ROW==0)?(0):(1)))
#define COL_BLOCKS (((NYPROB-2)/THREADS_PER_ROW)+(((NYPROB-2)%THREADS_PER_ROW==0)?(0):(1)))
#define NUMBER_OF_BLOCKS (ROW_BLOCKS*COL_BLOCKS)
#define PARMS_CX 0.1f
#define PARMS_CY 0.1f

// function to get the index for a 2d array that is saved as an 1d array
#define index2D(i,j,col) ((i)*(col)+(j))

void inidat(int, int, float*), prtdat(int, int, float*, char*);

#ifdef CONVERGE
// kernel that helps reducing faster
__global__ void semi_reduce(int *conv_flags)
{
	for (unsigned int s = blockDim.x/2 ; s > 0 ; s>>=1)
	{
		if (threadIdx.x < s) conv_flags[threadIdx.x + blockIdx.x*blockDim.x] += conv_flags[threadIdx.x+s + blockIdx.x*blockDim.x];
		__syncthreads();
	}
	
}

template <unsigned int threads>
__global__ void heat(float *old_grid, float *new_grid, int *conv_flags)
{
	__shared__ int block_conv_flags[THREADS_PER_BLOCK];
	int my_x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int my_y = (blockIdx.y * blockDim.y) + threadIdx.y;
	// Increment each coordinate by 1. This is used in an attempt to save resources: 
	// The first row and column stay always 0, so there is no point in having threads there just sitting.
	my_x += 1;
	my_y += 1;


	//if (my_x < NXPROB-1 && my_y < NYPROB-1) grids[0][index2D(my_x,my_y,NYPROB)] = 1.0f*threadIdx.x + 0.01f*threadIdx.y; 	//DEBUG
	
	// update cell
	if (my_x < NXPROB-1 && my_y < NYPROB-1)
	{
		new_grid[index2D(my_x,my_y,NYPROB)] = old_grid[index2D(my_x,my_y,NYPROB)] + 
							PARMS_CX * (old_grid[index2D(my_x+1,my_y,NYPROB)] + 
							old_grid[index2D(my_x-1,my_y,NYPROB)] - 
							2.0f * old_grid[index2D(my_x,my_y,NYPROB)]) + 
							PARMS_CY * (old_grid[index2D(my_x,my_y+1,NYPROB)] + 
							old_grid[index2D(my_x,my_y-1,NYPROB)] - 
							2.0f * old_grid[index2D(my_x,my_y,NYPROB)]);
		// check for convergence of my element & update my flag
		if ((float)fabs(old_grid[index2D(my_x,my_y,NYPROB)] - new_grid[index2D(my_x,my_y,NYPROB)]) < 1e-3f)
			block_conv_flags[index2D(threadIdx.x,threadIdx.y,blockDim.x)] = 1;
		else
			block_conv_flags[index2D(threadIdx.x,threadIdx.y,blockDim.x)] = 0;
	}
	else
		block_conv_flags[index2D(threadIdx.x,threadIdx.y,blockDim.x)] = 1;

	// syncrhonize threads to ensure that all data is updated before checking for convergence
	__syncthreads();
	
	// check for convergence of the block
	// not so efficient implementation
/*	for (unsigned int i = blockDim.y/2 ; i > 0 ; i>>=1)
	{
		if (threadIdx.y < i) block_conv_flags[index2D(threadIdx.x,threadIdx.y,blockDim.x)] += block_conv_flags[index2D(threadIdx.x,threadIdx.y+i,blockDim.x)];
		__syncthreads();
	}
	if (threadIdx.y == 0)
	{
		for (unsigned int i = blockDim.x/2 ; i > 0 ; i>>=1)
		{
			if (threadIdx.x < i) block_conv_flags[index2D(threadIdx.x,threadIdx.y,blockDim.x)] += block_conv_flags[index2D(threadIdx.x+i,threadIdx.y,blockDim.x)];
			__syncthreads();
		}
	}
	if (block_conv_flags[0] == THREADS_PER_BLOCK)
		conv_flags[index2D(blockIdx.x,blockIdx.y,COL_BLOCKS)] = 1;
	else
		conv_flags[index2D(blockIdx.x,blockIdx.y,COL_BLOCKS)] = 0;
*/
	// more efficient implementation
	// block is fix-sized, so the loop is unrolled for greater efficiency:
	// add the upper halfs of the matrix "recursively"
	// when only one line is left, add the left halfs "recursively".
	// case 4096 threads (64x64)
	if (threads == 4096) { if (threadIdx.y < 32) block_conv_flags[index2D(threadIdx.x,threadIdx.y,blockDim.x)] += block_conv_flags[index2D(threadIdx.x,threadIdx.y+32,blockDim.x)]; __syncthreads(); }
	// case 1024 threads (32x32)
	if (threads >= 1024) { if (threadIdx.y < 16) block_conv_flags[index2D(threadIdx.x,threadIdx.y,blockDim.x)] += block_conv_flags[index2D(threadIdx.x,threadIdx.y+16,blockDim.x)]; __syncthreads(); }
	// case 526 threads (16x16)
	if (threads >= 256) { if (threadIdx.y < 8) block_conv_flags[index2D(threadIdx.x,threadIdx.y,blockDim.x)] += block_conv_flags[index2D(threadIdx.x,threadIdx.y+8,blockDim.x)]; __syncthreads(); }
	if (threads >= 64) { if (threadIdx.y < 4) block_conv_flags[index2D(threadIdx.x,threadIdx.y,blockDim.x)] += block_conv_flags[index2D(threadIdx.x,threadIdx.y+4,blockDim.x)];__syncthreads();}

	if (threadIdx.y < 2) block_conv_flags[index2D(threadIdx.x,threadIdx.y,blockDim.x)] += block_conv_flags[index2D(threadIdx.x,threadIdx.y+2,blockDim.x)]; 
	__syncthreads();
	if (threadIdx.y == 0) block_conv_flags[index2D(threadIdx.x,threadIdx.y,blockDim.x)] += block_conv_flags[index2D(threadIdx.x,threadIdx.y+1,blockDim.x)]; 
	__syncthreads();

	// only one line is left
	if (threadIdx.y == 0)
	{
		// case 4096 threads (64x64)
		if (threads >= 4096) { if (threadIdx.x < 32) block_conv_flags[index2D(threadIdx.x,threadIdx.y,blockDim.x)] += block_conv_flags[index2D(threadIdx.x+32,threadIdx.y,blockDim.x)]; __syncthreads(); }
		// case 1024 threads (32x32)
		if (threads >= 1024) { if (threadIdx.x < 16) block_conv_flags[index2D(threadIdx.x,threadIdx.y,blockDim.x)] += block_conv_flags[index2D(threadIdx.x+16,threadIdx.y,blockDim.x)]; __syncthreads(); }
		// case 526 threads (16x16)
		if (threads >= 256) { if (threadIdx.x < 8) block_conv_flags[index2D(threadIdx.x,threadIdx.y,blockDim.x)] += block_conv_flags[index2D(threadIdx.x+8,threadIdx.y,blockDim.x)]; __syncthreads(); }
		if (threads >= 64) { if (threadIdx.x < 4) block_conv_flags[index2D(threadIdx.x,threadIdx.y,blockDim.x)] += block_conv_flags[index2D(threadIdx.x+4,threadIdx.y,blockDim.x)]; __syncthreads(); }

		if (threadIdx.x < 2) block_conv_flags[index2D(threadIdx.x,threadIdx.y,blockDim.x)] += block_conv_flags[index2D(threadIdx.x+2,threadIdx.y,blockDim.x)]; 
		__syncthreads();
		if (threadIdx.x == 0)
		{
			block_conv_flags[index2D(threadIdx.x,threadIdx.y,blockDim.x)] += block_conv_flags[index2D(threadIdx.x+1,threadIdx.y,blockDim.x)];
			// check if block has converged
			if (block_conv_flags[0] == threads)
				conv_flags[index2D(blockIdx.x,blockIdx.y,COL_BLOCKS)] = 1;
			else
				conv_flags[index2D(blockIdx.x,blockIdx.y,COL_BLOCKS)] = 0;
		}
	}
}
#else
__global__ void heat(float *old_grid, float *new_grid)
{
        int my_x = (blockIdx.x * blockDim.x) + threadIdx.x;
        int my_y = (blockIdx.y * blockDim.y) + threadIdx.y;
        // Increment each coordinate by 1. This is used in an attempt to save resources: 
        // The first row and column stay always 0, so there is no point in having threads there just sitting.
        my_x += 1;
        my_y += 1;


        //if (my_x < NXPROB-1 && my_y < NYPROB-1) grids[0][index2D(my_x,my_y,NYPROB)] = 1.0f*threadIdx.x + 0.01f*threadIdx.y;   //DEBUG

        // update cell
        if (my_x < NXPROB-1 && my_y < NYPROB-1)
        {
                new_grid[index2D(my_x,my_y,NYPROB)] = old_grid[index2D(my_x,my_y,NYPROB)] +
                                                        PARMS_CX * (old_grid[index2D(my_x+1,my_y,NYPROB)] +
                                                        old_grid[index2D(my_x-1,my_y,NYPROB)] -
                                                        2.0f * old_grid[index2D(my_x,my_y,NYPROB)]) +
                                                        PARMS_CY * (old_grid[index2D(my_x,my_y+1,NYPROB)] +
                                                        old_grid[index2D(my_x,my_y-1,NYPROB)] -
                                                        2.0f * old_grid[index2D(my_x,my_y,NYPROB)]);
        }
}
#endif

int main(void)
{
	int i, old = 0;
	float *grid, msecs;	// arrays for grids (allocated dynamically, so stack won't get smashed in case of large input
	if ((grid = (float *)malloc(NXPROB*NYPROB*sizeof(float))) == NULL)
	{
		perror("malloc for grid");
		return -1;
	}

	// allocate space in device for the 2 arrays
	float *dvc_grid0, *dvc_grid1, *dvc_grids[2];
	cudaMalloc((void**)&dvc_grid0, NXPROB*NYPROB*sizeof(float));
	cudaMalloc((void**)&dvc_grid1, NXPROB*NYPROB*sizeof(float));

#ifdef CONVERGE
	// allocate space for convergence flags. Last element ("+1") serves as the the total convergence flag
	int conv_flag = 0, *dvc_conv_flags, temp, blocks;
	cudaMalloc((void**)&dvc_conv_flags, NUMBER_OF_BLOCKS*sizeof(int));
	// initialize total convergence flag to 0
	conv_flag = 0;
	cudaMemcpy(dvc_conv_flags+NUMBER_OF_BLOCKS, &conv_flag, sizeof(int), cudaMemcpyHostToDevice);
#endif

	dvc_grids[0] = dvc_grid0;
	dvc_grids[1] = dvc_grid1;

	// initialize starting grid
	inidat(NXPROB, NYPROB, grid);

	// copy the initialized data to device memory
	cudaMemcpy(dvc_grid0, grid, NXPROB*NYPROB*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dvc_grid1, grid, NXPROB*NYPROB*sizeof(float), cudaMemcpyHostToDevice);

	dim3 numofthreads(THREADS_PER_ROW, THREADS_PER_ROW);
	dim3 numofblocks(ROW_BLOCKS, COL_BLOCKS);
//	printf("%d %d %d %d \n", THREADS_PER_BLOCK, NUMBER_OF_BLOCKS, COL_BLOCKS, ROW_BLOCKS);	//DEBUG
	timestamp t_start = getTimestamp();
	for (i = 0 ; i < STEPS ; ++i)
	{
#ifndef CONVERGE
		// launch kernel on GPU
		heat<<<numofblocks, numofthreads>>>(dvc_grids[old], dvc_grids[1-old]);

		// iteration finished, set current grid as old
		old = 1-old;
#else
		// launch heat kernel on GPU
		heat<THREADS_PER_BLOCK><<<numofblocks, numofthreads>>>(dvc_grids[old], dvc_grids[1-old], dvc_conv_flags);

		// iteration finished, set current grid as old
		old = 1-old;

		if (i % CHECK_INTERVAL == 0)
		{
			// check for convergence
			// launch semi-reduce kernel on gpu
			blocks = (NUMBER_OF_BLOCKS/THREADS_PER_BLOCK)+(NUMBER_OF_BLOCKS%THREADS_PER_BLOCK==0?0:1);
			semi_reduce<<<blocks, THREADS_PER_BLOCK>>>(dvc_conv_flags);

			// get the total convergence:
			// semi-reduce has created local totals in dvc_conv_flags array
			// Adding them creates the final convergence total
			for(int k = 0 ; k < blocks ; k++)
			{
				cudaMemcpy(&temp, dvc_conv_flags+k*THREADS_PER_BLOCK, sizeof(int), cudaMemcpyDeviceToHost);
				conv_flag += temp;
			};
			if (conv_flag == NUMBER_OF_BLOCKS) break;
			else conv_flag = 0;
		}
#endif
	}
	msecs = getElapsedtime(t_start);

	// get the results from GPU ("old" grid has the final values, after the last iteration)
	cudaMemcpy(grid, dvc_grids[old], NXPROB*NYPROB*sizeof(float), cudaMemcpyDeviceToHost);

	// output the results to file
	char buf[256];
#ifdef CONVEGRE
	sprintf(buf, "out_cuda_%d_%d_%d_CONVERGENCE.dat", THREADS_PER_BLOCK, NUMBER_OF_BLOCKS, STEPS);
#else
	sprintf(buf, "out_cuda_%d_%d_%d.dat", THREADS_PER_BLOCK, NUMBER_OF_BLOCKS, STEPS);
#endif
	prtdat(NXPROB, NYPROB, grid, buf);

	// print info
#ifdef CONVERGE
	if (conv_flag)
		printf("Converged at %d steps\n", i);
	else
		printf("Did not converge\n");
#endif
	printf("Elapsed time: %.3f %ssecs\n", (msecs/1000 > 1.0 ? msecs/1000 : msecs), (msecs/1000 > 1.0 ? "" : "m"));

	// free resources and exit
	cudaFree(dvc_grid0);
	cudaFree(dvc_grid1);
	cudaFree(dvc_grids);
	free(grid);

	return 0;
}

/*****************************************************************************
 *  *  subroutine inidat
 *   *****************************************************************************/
void inidat(int nx, int ny, float *u) {
int ix, iy;

for (ix = 0; ix <= nx-1; ix++)
  for (iy = 0; iy <= ny-1; iy++)
     *(u+ix*ny+iy) = (float)(ix * (nx - ix - 1) * iy * (ny - iy - 1));
}

/**************************************************************************
 *  * subroutine prtdat
 *   **************************************************************************/
void prtdat(int nx, int ny, float *unew, char *fnam) {
int ix, iy;
FILE *fp;

fp = fopen(fnam, "w");
for (iy = ny-1; iy >= 0; iy--) {
  for (ix = 0; ix <= nx-1; ix++) {
    fprintf(fp, "%6.1f", *(unew+ix*ny+iy));
    if (ix != nx-1)
      fprintf(fp, " ");
    else
      fprintf(fp, "\n");
    }
  }
fclose(fp);
}

