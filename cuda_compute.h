int CudaCorrelateLine(float* input_data, float* output_data, size_t numpy_row_stride, int rows, int cols);
int CudaReprojectToPolar(float *input_data, size_t input_row_stride, float *polar_data, size_t polar_row_stride,
                         int rows, int cols, int r_min, int r_max, int polar_angles,
                         float center_y, float center_x, float cval);
int CudaReprojectAndCorrelate(float* input_data, size_t input_row_stride, float* ccf_data, float* rad_data,
                              int rows, int cols, int r_min, int r_max, int polar_angles,
                              float center_y, float center_x, float cval);
int CudaReprojectAndCorrelateArray(float* input_data, int num_images, size_t input_image_stride, size_t input_row_stride,
                                   float* ccf_data, size_t ccf_row_stride,
                                   float* rad_data, size_t rad_row_stride,
                                   int rows, int cols, int r_min, int r_max, int polar_angles,
                                   float center_y, float center_x, float cval);
