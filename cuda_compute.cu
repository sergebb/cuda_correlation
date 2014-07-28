#include "cuda_compute.h"
#include <stdio.h>
 
#define BLOCK_SIZE 512

__global__ 
void gCorrelationCompute( float *dev_input, float *dev_output, int rows, int cols, size_t pitch) 
{
    // blockIdx.x; 
    // blockIdx.y;               
    // threadIdx.x; 
    int xCoord = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int yCoord = blockIdx.y;

    __shared__ float sumBuf[BLOCK_SIZE];
    sumBuf[tid] = 0;
    int i,j,p;
    unsigned int s;
    float val;
    float *t;
    if( xCoord<cols-1 && yCoord<rows){
        val = *((float*)((char*)dev_input + yCoord * pitch) + xCoord);  //Value in memory cell (xCoord,yCoord) in first line, const for every thread
        for( j=yCoord; j<rows; j++ ){
            for( i=0; i<cols; i++ ){
                p = (i+xCoord)%cols;                                            //Coordinate in shifted line(second line)
                t = (float*)((char*)dev_input + j * pitch) + p;            //Value in memory cell (p,yCoord) multiply to (xCoord,yCoord)
                sumBuf[tid] = *t * val;                                 //Compute per element multiplication

                for( s=1; s<blockDim.x; s*=2) {         //Sum reduction
                    if(tid % (2*s) == 0) {
                        sumBuf[tid] += sumBuf[tid + s];
                    }
                    __syncthreads();
                }
                // write result for this block to global mem
                if(tid == 0) {
                    t = (float*)((char*)dev_output + yCoord * pitch) + i; //memory cell (i,yCoord) where results of correlation is saved
                    atomicAdd(t, sumBuf[0]);      //Results is sum of I(x)I(x+I) averaged for all x, this block calculate 512 x'es, other added from other blocks (that is why +=) 
                    if(j!=yCoord){
                        t = (float*)((char*)dev_output + j * pitch) + i;    //Add result to j(first) line and yCoord(second) line, 1to2 and 2to1 correlation results 
                        atomicAdd(t, sumBuf[0]);                            //are the same, so we save some time by fixind j>=yCoord and do not calculate it again.
                    }
                }
            }
        }
    }
}
 
int CudaCorrelate( float** input_data, float** output_data, int rows, int cols)
{
    float *dev_input,*dev_output,*t;
    cudaError_t err;
    size_t pitch;
    //Memory allocation for input and output arrays
    err = cudaMallocPitch((void**)&dev_input, &pitch, sizeof(float)*cols, rows);  
    if( err != cudaSuccess ){
        fprintf( stderr, "Error at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    } 
    err = cudaMallocPitch((void**)&dev_output, &pitch, sizeof(float)*cols, rows);  
    if( err != cudaSuccess ){
        fprintf( stderr, "Error at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    } 
    //Memory copying&initialisation for input data and error-check
    for( int i=0; i<rows; i++ ){
        t = (float*)((char*)dev_input + i * pitch);// + Column;
        cudaMemcpy( t, input_data[i], pitch, cudaMemcpyHostToDevice );
    }
    for( int i=0; i<rows; i++ ){
        t = (float*)((char*)dev_output + i * pitch);// + Column;
        cudaMemset( t, 0, pitch);
    }
    err = cudaGetLastError();
    if( err != cudaSuccess ){
        fprintf( stderr, "Error at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    //Calculation    
    dim3 dimBlock( BLOCK_SIZE, 1 );
    dim3 dimGrid( cols/BLOCK_SIZE + ((cols%BLOCK_SIZE==0)?0:1), rows );
    printf("block(%d,%d) and grid(%d,%d), cols=%d, rows=%d\n",dimBlock.x,dimBlock.y,dimGrid.x,dimGrid.y,cols,rows);
    gCorrelationCompute<<<dimGrid,dimBlock>>>( dev_input, dev_output, rows, cols, pitch );
    // hello<<<dimGrid, dimBlock>>>(ad, bd);
    err = cudaDeviceSynchronize();
    if( err != cudaSuccess ){
        fprintf( stderr, "Error at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf ("Kernel execution time: %f ms\n", time);


    //Result memory copying back
    for( int i=0; i<rows; i++ ){
        t = (float*)((char*)dev_output + i * pitch);// + Column;
        cudaMemcpy( output_data[i], t, cols*sizeof(float), cudaMemcpyDeviceToHost );
    }
    err = cudaGetLastError();
    if( err != cudaSuccess ){
        fprintf( stderr, "Error at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    cudaFree( dev_input );
    cudaFree( dev_output );
    
    return EXIT_SUCCESS;
}