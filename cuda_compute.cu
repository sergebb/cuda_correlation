#include "cuda_compute.h"
#include <stdio.h>
 
#define BLOCK_SIZE 512

__global__ 
void gCorrelationComputeFull( float *dev_input, float *dev_output, int rows, int cols, size_t pitch) 
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
    if( xCoord<cols && yCoord<rows){
        val = *((float*)((char*)dev_input + yCoord * pitch) + xCoord);  //Value in memory cell (xCoord,yCoord) in first line, const for every thread
        for( j=yCoord; j<rows; j++ ){
            for( i=0; i<cols; i++ ){        //Loop over circular shifts
                p = (i+xCoord)%cols;                                            //Coordinate in shifted line(second line)
                t = (float*)((char*)dev_input + j * pitch) + p;            //Value in memory cell (p,yCoord) multiply to (xCoord,yCoord)
                sumBuf[tid] = *t * val;                                 //Compute per element multiplication

                // eff_len += (*t)*val>0 ? 1 : 0;  // pixel with 0 value considered as masked

                for( s=1; s<blockDim.x; s*=2) {         //Sum reduction
                    if(tid % (2*s) == 0) {
                        sumBuf[tid] += sumBuf[tid + s];
                    }
                    __syncthreads();
                }
                // write result for this block to global mem
                if(tid == 0) {
                    // sumBuf /= eff_len;      //Normalisation
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

__global__ 
void gCorrelationComputeLine( float *dev_input, float *dev_output, int *dev_norm, int rows, int cols, size_t pitchF, size_t pitchI) 
{
    // blockIdx.x; 
    // blockIdx.y;               
    // threadIdx.x; 
    int xCoord = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int yCoord = blockIdx.y;

    __shared__ float sumBuf[BLOCK_SIZE];
    __shared__ int effLenBuf[BLOCK_SIZE];
    sumBuf[tid] = 0;
    effLenBuf[tid] = 0;
    int i,pos;
    unsigned int s;
    float val;
    float *t;
    int *p;
    if( xCoord<cols && yCoord<rows){
        val = *((float*)((char*)dev_input + yCoord * pitchF) + xCoord);  //Value in memory cell (xCoord,yCoord) in first line, const for every thread
        for( i=0; i<cols; i++ ){        //Loop over circular shifts
            pos = (i+xCoord)%cols;                                            //Coordinate in shifted line(second line)
            t = (float*)((char*)dev_input + yCoord * pitchF) + pos;            //Value in memory cell (p,yCoord) multiply to (xCoord,yCoord)
            sumBuf[tid] = *t * val;                                 //Compute per element multiplication

            effLenBuf[tid] = (*t)*val>0 ? 1 : 0;  // pixel with 0 value considered as masked

            for( s=1; s<blockDim.x; s*=2) {         //Sum reduction
                if(tid % (2*s) == 0) {
                    sumBuf[tid] += sumBuf[tid + s];
                    effLenBuf[tid] += effLenBuf[tid + s];
                }
                __syncthreads();
            }
            // write result for this block to global mem
            if(tid == 0) {

                // sumBuf /= eff_len;      //Normalisation
                t = (float*)((char*)dev_output + yCoord * pitchF) + i; //memory cell (i,yCoord) where results of correlation is saved
                p = (int*)((char*)dev_norm + yCoord * pitchI) + i; //memory cell (i,yCoord) where results of correlation is saved
                atomicAdd(t, sumBuf[0]);      //Results is sum of I(x)I(x+I) averaged for all x, this block calculate 512 x'es, other added from other blocks (that is why +=) 
                atomicAdd(p, effLenBuf[0]);      //Results is sum of I(x)I(x+I) averaged for all x, this block calculate 512 x'es, other added from other blocks (that is why +=) 
            }
        }
    }
}

__global__ 
void gApplyNorm( float *dev_output, int *dev_norm, int rows, int cols, size_t pitchF, size_t pitchI) 
{
    // blockIdx.x; 
    // blockIdx.y;               
    // threadIdx.x; 
    int xCoord = blockIdx.x*blockDim.x + threadIdx.x;
    int yCoord = blockIdx.y;

    float *p;
    int *norm;

    if( xCoord<cols && yCoord<rows){
        p = (float*)((char*)dev_output + yCoord * pitchF) + xCoord;  
        norm = (int*)((char*)dev_norm + yCoord * pitchI) + xCoord;  
        if( *norm == 0 ) *norm = 1;
        *p = *p / *norm;
    }
}
 
int CudaCorrelateFull( float** input_data, float** output_data, int rows, int cols)
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

    // cudaEvent_t start, stop;
    // float time;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEventRecord(start, 0);

    //Calculation    
    dim3 dimBlock( BLOCK_SIZE, 1 );
    dim3 dimGrid( cols/BLOCK_SIZE + ((cols%BLOCK_SIZE==0)?0:1), rows );
    // printf("block(%d,%d) and grid(%d,%d), cols=%d, rows=%d\n",dimBlock.x,dimBlock.y,dimGrid.x,dimGrid.y,cols,rows);
    gCorrelationComputeFull<<<dimGrid,dimBlock>>>( dev_input, dev_output, rows, cols, pitch );
    // hello<<<dimGrid, dimBlock>>>(ad, bd);
    err = cudaDeviceSynchronize();
    if( err != cudaSuccess ){
        fprintf( stderr, "Error at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&time, start, stop);
    // printf ("Kernel execution time: %f ms\n", time);


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

int CudaCorrelateLine( float** input_data, float** output_data, int rows, int cols)
{
    float *dev_input,*dev_output,*t;
    int *dev_norm, *p;
    cudaError_t err;
    size_t pitchF, pitchI;
    //Memory allocation for input and output arrays
    err = cudaMallocPitch((void**)&dev_input, &pitchF, sizeof(float)*cols, rows);  
    if( err != cudaSuccess ){
        fprintf( stderr, "Error at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    } 
    err = cudaMallocPitch((void**)&dev_output, &pitchF, sizeof(float)*cols, rows);  
    if( err != cudaSuccess ){
        fprintf( stderr, "Error at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    } 
    err = cudaMallocPitch((void**)&dev_norm, &pitchI, sizeof(int)*cols, rows);  
    if( err != cudaSuccess ){
        fprintf( stderr, "Error at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    } 
    //Memory copying&initialisation for input data and error-check
    for( int i=0; i<rows; i++ ){
        t = (float*)((char*)dev_input + i * pitchF);// + Column;
        cudaMemcpy( t, input_data[i], pitchF, cudaMemcpyHostToDevice );
    }
    for( int i=0; i<rows; i++ ){
        t = (float*)((char*)dev_output + i * pitchF);// + Column;
        cudaMemset( t, 0, pitchF);
    }
    for( int i=0; i<rows; i++ ){
        p = (int*)((char*)dev_norm + i * pitchI);// + Column;
        cudaMemset( p, 0, pitchI);
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
    // printf("block(%d,%d) and grid(%d,%d), cols=%d, rows=%d\n",dimBlock.x,dimBlock.y,dimGrid.x,dimGrid.y,cols,rows);
    gCorrelationComputeLine<<<dimGrid,dimBlock>>>( dev_input, dev_output, dev_norm, rows, cols, pitchF, pitchI );
    gApplyNorm<<<dimGrid,dimBlock>>>( dev_output, dev_norm, rows, cols, pitchF, pitchI );
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
        t = (float*)((char*)dev_output + i * pitchF);// + Column;
        cudaMemcpy( output_data[i], t, cols*sizeof(float), cudaMemcpyDeviceToHost );
    }
    err = cudaGetLastError();
    if( err != cudaSuccess ){
        fprintf( stderr, "Error at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    cudaFree( dev_input );
    cudaFree( dev_output );
    cudaFree( dev_norm );
    
    return EXIT_SUCCESS;
}