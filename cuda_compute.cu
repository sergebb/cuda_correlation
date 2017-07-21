#include "cuda_compute.h"
#include <stdio.h>
#include "math_constants.h"
 
#define BLOCK_SIZE 512
#define MAX_WIDTH 2048

texture<float, 2 > inputDataTexRef;

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__
void gComputePolarProjection(float *dev_output, size_t pitchPolar, int image_number,
                             int rows, int cols, int r_min, int r_max, int polar_angles,
                             float center_y, float center_x, float cval)
{
    unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int t = blockIdx.y * blockDim.y + threadIdx.y;
    
    float radius = r_min + r;
    float angle_step = 2*CUDART_PI_F/polar_angles;
    float angle = -CUDART_PI_F + t*angle_step ;
    float res;

    if(radius < r_max && t < polar_angles){
        float x = radius * cos(angle) + center_x + 0.5f;
        float y = radius * sin(angle) + center_y + 0.5f;
        if(x<0 || x>= cols || y<0 || y>= rows){
            res = cval;
        }else{
            res = tex2D(inputDataTexRef, x, y + image_number*rows);
        }
        *((float*)((char*)dev_output + r * pitchPolar) + t) = res;
    }
}


__global__ void gCorrelationComputeLine( float *dev_input, float *dev_output, int rows, int cols, size_t pitchInput) 
{
    int xCoord = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int yCoord = blockIdx.y;

    __shared__ float line_data[MAX_WIDTH];
    int i,pos;
    float val;
    float t;
    float sum = 0;
    if( xCoord<cols && yCoord<rows){
        for(i=0; i*BLOCK_SIZE < cols; i++){
            if(tid+i*BLOCK_SIZE < cols){
                line_data[tid + i*BLOCK_SIZE] = *((float*)((char*)dev_input + yCoord * pitchInput) + tid + i*BLOCK_SIZE); //Copy line of matrix to shared memory
            }
        }
        __syncthreads();

        for( i=0; i<cols; i++ ){        //Loop over whole line
            val = line_data[i];         //First multiplier
            pos = (i+xCoord)%cols;      //Coordinate in shifted line(second line)
            t = line_data[pos];         //Value in memory cell (pos,yCoord), second multiplier
            sum += t * val;             //Sum of multiplications
        }
        sum /= cols;    //Divide by line len -> correlation
        
        *((float*)((char*)dev_output + yCoord * pitchInput) + xCoord) = sum; //memory cell (xCoord,yCoord) where results of correlation is saved
    }
}
    
__global__ 
void gRecoverMask( float *dev_input, int rows, int cols, size_t pitchInput) 
{
    int xCoord = threadIdx.x;
    int yCoord = blockIdx.y;
    int i,s;
    float val;
    float average_val;    
    
    float sum_value = 0;
    int non_mask = 0;
    __shared__ float sum_buf[BLOCK_SIZE];
    __shared__ float non_mask_buf[BLOCK_SIZE];

    for(i=0; i*BLOCK_SIZE < cols; i++){
        if(xCoord + i*BLOCK_SIZE < cols){
            val = *((float*)((char*)dev_input + yCoord * pitchInput) + xCoord + i*BLOCK_SIZE);
        }else{
            val = -1;
        }

        if(val >= 0){
            non_mask = 1;
        }else{
            val = 0;
            non_mask = 0;
        }
        
        sum_buf[xCoord] = val;
        non_mask_buf[xCoord] = non_mask;
        for( s=1; s<BLOCK_SIZE; s*=2) {         //Sum reduction
            if(xCoord % (2*s) == 0) {
                sum_buf[xCoord] += sum_buf[xCoord + s];
                non_mask_buf[xCoord] += non_mask_buf[xCoord + s];
            }
            __syncthreads();
        }
        if(xCoord == 0) {
            sum_value += sum_buf[0];
            non_mask += non_mask_buf[0];
        }
    }
    if(xCoord == 0) {
        sum_buf[0] = sum_value;
        non_mask_buf[0] = non_mask;
    }
    __syncthreads();
    average_val = sum_buf[0]/non_mask_buf[0];
    
    for(i=0; i*BLOCK_SIZE < cols; i++){
        if(xCoord + i*BLOCK_SIZE < cols){
            val = *((float*)((char*)dev_input + yCoord * pitchInput) + xCoord + i*BLOCK_SIZE);
            if(val < 0){
                *((float*)((char*)dev_input + yCoord * pitchInput) + xCoord + i*BLOCK_SIZE) = average_val;
            }
        }
    }
}
__global__
void gCCFAngle(float *dev_ccf_2d, float *dev_ccf_angle, int radius_range, size_t pitchF){
    int xCoord = blockIdx.x;
    int yCoord = threadIdx.y;
    int i,s;
    __shared__ float sum_buf[BLOCK_SIZE];
    float sum_value = 0;
    
    for(i=0; i*BLOCK_SIZE < radius_range; i++){
        if(yCoord + i*BLOCK_SIZE < radius_range){
            sum_buf[yCoord] = *((float*)((char*)dev_ccf_2d + (yCoord + i*BLOCK_SIZE) * pitchF) + xCoord);
        }else{
            sum_buf[yCoord] = 0;
        }
        for( s=1; s<BLOCK_SIZE; s*=2) {         //Sum reduction
            if(yCoord % (2*s) == 0) {
                sum_buf[yCoord] += sum_buf[yCoord + s];
            }
            __syncthreads();
        }
        if(yCoord == 0){
            sum_value += sum_buf[0];
        }
    }
    if(yCoord == 0){
        dev_ccf_angle[xCoord] = sum_value/radius_range;
    }
    
}


int CudaReprojectToPolar(float *input_data, size_t input_row_stride, float *polar_data, size_t polar_row_stride,
                         int rows, int cols, int r_min, int r_max, int polar_angles,
                         float center_y, float center_x, float cval)
{
    float *dev_input, *dev_output;
    size_t pitchInput, pitchPolar;
    
    // ///////////////////////
    // cudaEvent_t start, stop;
    // float time;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start, 0);
    // ///////////////////////

    //Memory allocation for input and output arrays
    CUDA_CHECK(cudaMallocPitch((void**)&dev_input, &pitchInput, sizeof(float)*cols, rows));
    CUDA_CHECK(cudaMallocPitch((void**)&dev_output, &pitchPolar, sizeof(float)*polar_angles, r_max-r_min));

    // CUDA_CHECK(cudaMemcpy(dev_input, input_data, rows*input_row_stride, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(dev_input, pitchInput, input_data, input_row_stride, sizeof(float)*cols, rows, cudaMemcpyHostToDevice));

    // Specify texture    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    CUDA_CHECK(cudaBindTexture2D(NULL, inputDataTexRef, dev_input, channelDesc, cols, rows, pitchInput));
    inputDataTexRef.addressMode[0] = cudaAddressModeBorder;
    inputDataTexRef.addressMode[1] = cudaAddressModeBorder;
    inputDataTexRef.filterMode = cudaFilterModeLinear;
    inputDataTexRef.normalized = false;
    
    //Calculation    
    dim3 projBlock( 32, 32 );
    dim3 projGrid((r_max - r_min + projBlock.y - 1) / projBlock.y, 
                  (polar_angles + projBlock.x - 1) / projBlock.x);
    
    gComputePolarProjection<<<projGrid,projBlock>>>(dev_output, pitchPolar, 0,
                                                    rows, cols, r_min, r_max, polar_angles,
                                                    center_y, center_x, cval);
    
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy2D(polar_data, polar_row_stride, dev_output, pitchPolar, sizeof(float)*polar_angles, r_max-r_min, cudaMemcpyDeviceToHost));

    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&time, start, stop);
    // printf ("CUDA execution time: %f ms\n", time);

    cudaFree( dev_input );
    cudaFree( dev_output );

    return EXIT_SUCCESS;
}

int CudaCorrelateLine(float* input_data, float* output_data, size_t numpy_row_stride, int rows, int cols)
{
    float *dev_input, *dev_output;
    size_t pitchInput;
    
    if( cols > MAX_WIDTH ){
        fprintf( stderr, "Error at %s:%i : %s\n", __FILE__, __LINE__, "Image width exceeds max value, need recompile" );
        exit( -1 );
    } 
    
    //Memory allocation for input and output arrays
    CUDA_CHECK(cudaMallocPitch((void**)&dev_input, &pitchInput, sizeof(float)*cols, rows));
    CUDA_CHECK(cudaMallocPitch((void**)&dev_output, &pitchInput, sizeof(float)*cols, rows));

    CUDA_CHECK(cudaMemcpy2D(dev_input, pitchInput, input_data, numpy_row_stride, sizeof(float)*cols, rows, cudaMemcpyHostToDevice));

    //RecoverMask
    dim3 recoverBlock( BLOCK_SIZE, 1 );
    dim3 recoverGrid( 1, rows );
    gRecoverMask<<<recoverGrid,recoverBlock>>>(dev_input, rows, cols, pitchInput);
    //Calculation
    dim3 corrBlock( BLOCK_SIZE, 1 );
    dim3 corrGrid( cols/BLOCK_SIZE + ((cols%BLOCK_SIZE==0)?0:1), rows );
    gCorrelationComputeLine<<<corrGrid,corrBlock>>>(dev_input, dev_output, rows, cols, pitchInput);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy2D(output_data, numpy_row_stride, dev_output, pitchInput, sizeof(float)*cols, rows, cudaMemcpyDeviceToHost));

    cudaFree(dev_input);
    cudaFree(dev_output);
    
    return EXIT_SUCCESS;
}

int CudaReprojectAndCorrelate(float* input_data, size_t input_row_stride, float* output_data, 
                              int rows, int cols, int r_min, int r_max, int polar_angles,
                              float center_y, float center_x, float cval)
{
    float *dev_input, *dev_polar_input, *dev_ccf_2d;
    float *dev_ccf_angle;
    size_t pitchInput, pitchPolar;
    
    // ///////////////////////
    // cudaEvent_t start, stop;
    // float time;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start, 0);
    // ///////////////////////

    //Memory allocation for input and output arrays
    CUDA_CHECK(cudaMallocPitch((void**)&dev_input, &pitchInput, sizeof(float)*cols, rows));
    CUDA_CHECK(cudaMallocPitch((void**)&dev_polar_input, &pitchPolar, sizeof(float)*polar_angles, r_max-r_min));
    CUDA_CHECK(cudaMallocPitch((void**)&dev_ccf_2d, &pitchPolar, sizeof(float)*polar_angles, r_max-r_min));
    CUDA_CHECK(cudaMalloc((void**)&dev_ccf_angle, sizeof(float)*polar_angles));

    //Memory copying&initialisation for input data and error-check
    CUDA_CHECK(cudaMemcpy2D(dev_input, pitchInput, input_data, input_row_stride, sizeof(float)*cols, rows, cudaMemcpyHostToDevice));

    //Calculation    
    dim3 projBlock( 32, 32 );
    dim3 projGrid((r_max - r_min + projBlock.y - 1) / projBlock.y, 
                  (polar_angles + projBlock.x - 1) / projBlock.x);

    // Specify texture    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    CUDA_CHECK(cudaBindTexture2D(NULL, inputDataTexRef, dev_input, channelDesc, cols, rows, pitchInput));
    inputDataTexRef.addressMode[0] = cudaAddressModeBorder;
    inputDataTexRef.addressMode[1] = cudaAddressModeBorder;
    inputDataTexRef.filterMode = cudaFilterModeLinear;
    inputDataTexRef.normalized = false;
    
    //Projection calculation
    gComputePolarProjection<<<projGrid,projBlock>>>(dev_polar_input, pitchPolar, 0,
                                                    rows, cols, r_min, r_max, polar_angles,
                                                    center_y, center_x, 0);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaUnbindTexture(inputDataTexRef));
    
    //RecoverMask
    dim3 recoverBlock( BLOCK_SIZE, 1 );
    dim3 recoverGrid( 1, r_max-r_min );
    gRecoverMask<<<recoverGrid,recoverBlock>>>( dev_polar_input, r_max-r_min, polar_angles, pitchPolar);

    dim3 corrBlock( BLOCK_SIZE, 1 );
    dim3 corrGrid( cols/BLOCK_SIZE + ((cols%BLOCK_SIZE==0)?0:1), rows );
    gCorrelationComputeLine<<<corrGrid,corrBlock>>>( dev_polar_input, dev_ccf_2d, r_max-r_min, polar_angles, pitchPolar);
    
    dim3 angleBlock( 1, BLOCK_SIZE );
    dim3 angleGrid( polar_angles, 1 );
    gCCFAngle<<<angleGrid,angleBlock>>>(dev_ccf_2d, dev_ccf_angle, r_max-r_min, pitchPolar);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy( output_data, dev_ccf_angle, polar_angles*sizeof(float), cudaMemcpyDeviceToHost ));

    // ///////////////////////////
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&time, start, stop);
    // printf ("CUDA total time: %f ms\n", time);
    // ///////////////////////////

    cudaFree(dev_input);
    cudaFree(dev_polar_input);
    cudaFree(dev_ccf_2d);
    cudaFree(dev_ccf_angle);

    return EXIT_SUCCESS;
}

int CudaReprojectAndCorrelateArray(float* input_data, int num_images, size_t input_image_stride, size_t input_row_stride,
                                   float* output_data, size_t output_row_stride,
                                   int rows, int cols, int r_min, int r_max, int polar_angles,
                                   float center_y, float center_x, float cval)
{
    float *dev_input, *dev_polar_input, *dev_ccf_2d;
    float *dev_ccf_angle;
    size_t pitchInput, pitchPolar;
    
    // ///////////////////////////
    // cudaEvent_t start, stop;
    // float time;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start, 0);
    // ///////////////////////////

    //Memory allocation for input and output arrays
    CUDA_CHECK(cudaMallocPitch((void**)&dev_input, &pitchInput, sizeof(float)*cols, rows*num_images));
    CUDA_CHECK(cudaMallocPitch((void**)&dev_polar_input, &pitchPolar, sizeof(float)*polar_angles, r_max-r_min));
    CUDA_CHECK(cudaMallocPitch((void**)&dev_ccf_2d, &pitchPolar, sizeof(float)*polar_angles, r_max-r_min));
    CUDA_CHECK(cudaMallocPitch((void**)&dev_ccf_angle, &pitchPolar, sizeof(float)*polar_angles, num_images));
    
    // Specify texture    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    CUDA_CHECK(cudaBindTexture2D(NULL, inputDataTexRef, dev_input, channelDesc, cols, rows*num_images, pitchInput));
    inputDataTexRef.addressMode[0] = cudaAddressModeBorder;
    inputDataTexRef.addressMode[1] = cudaAddressModeBorder;
    inputDataTexRef.filterMode = cudaFilterModeLinear;
    inputDataTexRef.normalized = false;

    //Memory copying&initialisation for input data and error-check
    CUDA_CHECK(cudaMemcpy2D(dev_input, pitchInput, input_data, input_row_stride, sizeof(float)*cols, rows*num_images, cudaMemcpyHostToDevice));

    for(int n=0; n<num_images; n++){
        //Calculation    
        dim3 projBlock( 32, 32 );
        dim3 projGrid((r_max - r_min + projBlock.y - 1) / projBlock.y, 
                    (polar_angles + projBlock.x - 1) / projBlock.x);

        //Projection calculation
        gComputePolarProjection<<<projGrid,projBlock>>>(dev_polar_input, pitchPolar, n,
                                                        rows, cols, r_min, r_max, polar_angles,
                                                        center_y, center_x, 0);
        
        //RecoverMask
        dim3 recoverBlock( BLOCK_SIZE, 1 );
        dim3 recoverGrid( 1, r_max-r_min );
        gRecoverMask<<<recoverGrid,recoverBlock>>>( dev_polar_input, r_max-r_min, polar_angles, pitchPolar);

        dim3 corrBlock( BLOCK_SIZE, 1 );
        dim3 corrGrid( cols/BLOCK_SIZE + ((cols%BLOCK_SIZE==0)?0:1), rows );
        gCorrelationComputeLine<<<corrGrid,corrBlock>>>( dev_polar_input, dev_ccf_2d, r_max-r_min, polar_angles, pitchPolar);
        
        dim3 angleBlock( 1, BLOCK_SIZE );
        dim3 angleGrid( polar_angles, 1 );
        gCCFAngle<<<angleGrid,angleBlock>>>(dev_ccf_2d, (float *)((char *)dev_ccf_angle + n*pitchPolar), r_max-r_min, pitchPolar);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy2D(output_data, output_row_stride, dev_ccf_angle, pitchPolar, polar_angles*sizeof(float), num_images, cudaMemcpyDeviceToHost));
    }

    // ///////////////////////////
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&time, start, stop);
    // printf ("CUDA total time: %f ms\n", time);
    // ///////////////////////////

    cudaFree(dev_input);
    cudaFree(dev_polar_input);
    cudaFree(dev_ccf_2d);
    cudaFree(dev_ccf_angle);

    return EXIT_SUCCESS;
}