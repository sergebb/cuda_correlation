int CudaCorrelateLine(float** input_data, float** mask_data,float** output_data, int rows, int cols);
int CudaReprojectToPolar(float **input_data, float **polar_data,
                         int rows, int cols, int r_min, int r_max, int polar_angles,
                         float center_y, float center_x, float cval);
int CudaReprojectAndCorrelate(float** input_data, float** mask_data,float* output_data, 
                              int rows, int cols, int r_min, int r_max, int polar_angles,
                              float center_y, float center_x, float cval);

