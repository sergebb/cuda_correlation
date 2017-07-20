#include <Python.h>
#include <string>
#include <boost/python.hpp>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <ctime>
#include <numpy/ndarrayobject.h> // ensure you include this header
#include "cuda_compute.h"
using namespace boost::python;

namespace { // Avoid cluttering the global namespace.

    void cComputePolarProjection(float **input_data, float **polar_data,
                                 unsigned int rows, unsigned int cols, unsigned int r_min, unsigned int r_max, unsigned int polar_angles,
                                 double center_y, double center_x, double cval){
        unsigned int n_rad = r_max - r_min;
        unsigned int n_angle = polar_angles;
        double angle_max = M_PI;
        double angle_min = -M_PI;
        
        double r_step = 1;
        double angle_step = 2*M_PI/n_angle;
        
        int i,j;
        int x_round, y_round;
        double r,t;
        double x,y;
        double a00,a01,a10,a11;
        double x_off,y_off;
        
        for(i = 0; i < n_rad; i++){
            r = r_min + i*r_step;
            for(j = 0; j < n_angle; j++){
                t = angle_min + (j+1)*angle_step;
                x = r * cos(t) + center_x;
                y = r * sin(t) + center_y;
                x_round = floor(x);
                y_round = floor(y);
                
                if(x_round>=0 && x_round<rows-1 && y_round>=0 && y_round<rows-1){
                    x_off = x-x_round;
                    y_off = y-y_round;
                    a00 = input_data[y_round][x_round];
                    a10 = input_data[y_round][x_round+1] - input_data[y_round][x_round];
                    a01 = input_data[y_round+1][x_round] - input_data[y_round][x_round];
                    a11 = input_data[y_round][x_round] + input_data[y_round+1][x_round+1] - (input_data[y_round+1][x_round] + input_data[y_round][x_round+1]);
                    
                    polar_data[i][j] = a00 + a10*x_off + a01*y_off + a11*x_off*y_off;
                }else{
                    polar_data[i][j] = cval;
                }
            }
        }
    }
    
    void ReprojectToPolar( boost::python::numeric::array& input_data, boost::python::numeric::array& polar_data, 
                           float center_y, float center_x, float r_min, float r_max, float cval )
    {
        PyArrayObject *ptr      = (PyArrayObject *) input_data.ptr();   //Get numpy data ptr
        PyArrayObject *polar_ptr  = (PyArrayObject *) polar_data.ptr();
        if (ptr == NULL) {
            std::cerr << "Could not get NP array." << std::endl;
            return;
        }
        if (polar_ptr == NULL) {
            std::cerr << "Could not get NP array." << std::endl;
            return;
        }
        const int dims     =  PyArray_NDIM(ptr);    //Get numpy array dimension
        const int new_dims =  PyArray_NDIM(polar_ptr);    //Get numpy array dimension
        if (dims != 2 || new_dims != 2){
            std::cerr << "Wrong dimension on array." << std::endl;
            return;
        }
        int rows = *(PyArray_DIMS(ptr));        //Get numpy array size
        int cols = *(PyArray_DIMS(ptr)+1);
        int polar_radii = *(PyArray_DIMS(polar_ptr));        //Get numpy array size
        int polar_angles = *(PyArray_DIMS(polar_ptr)+1);

        if (rows < 2 || cols < 2 || polar_radii < 2 || polar_angles < 2 ){
            std::cerr << "Arrays have wrong shape." << std::endl;
            return;
        }
        
        if (center_x < 0 || center_x > cols || center_y < 0 || center_y > rows ){
            std::cerr << "Center position outside the image" << std::endl;
            return;
        }

        if (ptr->descr->elsize != sizeof(float) //Test numpy array type
            || polar_ptr->descr->elsize != sizeof(float))
        {
            std::cerr << "Must be numpy.float32 ndarray" << std::endl;
            return;
        }
        
        
        if (r_max - r_min != polar_radii)
        {
            std::cerr << "polar_data shape[0] should be equal to " << (r_max - r_min) << std::endl;
            return;
        }

        float **finput = new float*[rows];
        float **foutput = new float*[rows];
        for ( int n=0; n<rows; n++ ){
            finput[n] = static_cast<float*> PyArray_GETPTR2(ptr, n, 0);
            foutput[n] = static_cast<float*> PyArray_GETPTR2(polar_ptr, n, 0);
        }

        CudaReprojectToPolar(finput, foutput, rows, cols, r_min, r_max, polar_angles, center_y, center_x, cval);

        delete[] finput;
        delete[] foutput;

        return;
    }
    
    void CorrealateLine( boost::python::numeric::array& data, boost::python::numeric::array& mask, boost::python::numeric::array& newdata )
    {
        PyArrayObject *ptr      = (PyArrayObject *) data.ptr();   //Get numpy data ptr
        PyArrayObject *mask_ptr = (PyArrayObject *) mask.ptr();
        PyArrayObject *new_ptr  = (PyArrayObject *) newdata.ptr();
        if (ptr == NULL) {
            std::cerr << "Could not get NP array." << std::endl;
            return;
        }
        if (mask_ptr == NULL) {
            std::cerr << "Could not get NP array." << std::endl;
            return;
        }
        if (new_ptr == NULL) {
            std::cerr << "Could not get NP array." << std::endl;
            return;
        }
        const int dims     =  PyArray_NDIM(ptr);    //Get numpy array dimension
        const int mask_dims     =  PyArray_NDIM(mask_ptr);    //Get numpy array dimension
        const int new_dims =  PyArray_NDIM(new_ptr);    //Get numpy array dimension
        if (dims != 2){
            std::cerr << "Wrong dimension on array." << std::endl;
            return;
        }
        if (dims != mask_dims || dims != new_dims){
            std::cerr << "Arrays have different shape." << std::endl;
            return;
        }
        int rows = *(PyArray_DIMS(ptr));        //Get numpy array size
        int cols = *(PyArray_DIMS(ptr)+1);
        int mask_rows = *(PyArray_DIMS(mask_ptr));        //Get numpy array size
        int mask_cols = *(PyArray_DIMS(mask_ptr)+1);
        int new_rows = *(PyArray_DIMS(new_ptr));        //Get numpy array size
        int new_cols = *(PyArray_DIMS(new_ptr)+1);

        if (rows != mask_rows || cols != mask_cols || rows != new_rows || cols != new_cols ){
            std::cerr << "Arrays have different shape." << std::endl;
            return;
        }

        if (ptr->descr->elsize != sizeof(float) //Test numpy array type
            || mask_ptr->descr->elsize != sizeof(float)
            || new_ptr->descr->elsize != sizeof(float))
        {
            std::cerr << "Must be numpy.float32 ndarray" << std::endl;
            return;
        }

        float **finput = new float*[rows];
        float **fmask = new float*[rows];
        float **foutput = new float*[rows];
        for ( int n=0; n<rows; n++ ){
            finput[n] = static_cast<float*> PyArray_GETPTR2(ptr, n, 0);
            fmask[n] = static_cast<float*> PyArray_GETPTR2(mask_ptr, n, 0);
            foutput[n] = static_cast<float*> PyArray_GETPTR2(new_ptr, n, 0);
        }

        CudaCorrelateLine(finput, fmask, foutput, rows, cols);

        delete[] finput;
        delete[] fmask;
        delete[] foutput;

        return;
    }
    void ReprojectAndCorrealate(boost::python::numeric::array& data, boost::python::numeric::array& mask, boost::python::numeric::array& ccf_data,
                                float center_y, float center_x, float r_min, float r_max, float cval )
    {
        PyArrayObject *ptr      = (PyArrayObject *) data.ptr();   //Get numpy data ptr
        PyArrayObject *mask_ptr = (PyArrayObject *) mask.ptr();
        PyArrayObject *ccf_ptr  = (PyArrayObject *) ccf_data.ptr();
        if (ptr == NULL) {
            std::cerr << "Could not get NP array." << std::endl;
            return;
        }
        if (mask_ptr == NULL) {
            std::cerr << "Could not get NP array." << std::endl;
            return;
        }
        if (ccf_ptr == NULL) {
            std::cerr << "Could not get NP array." << std::endl;
            return;
        }
        const int dims      = PyArray_NDIM(ptr);        //Get numpy array dimension
        const int mask_dims = PyArray_NDIM(mask_ptr);   //Get numpy array dimension
        const int new_dims  = PyArray_NDIM(ccf_ptr);    //Get numpy array dimension
        if (dims != 2){
            std::cerr << "Wrong dimension on array." << std::endl;
            return;
        }
        if (dims != mask_dims){
            std::cerr << "Arrays have different shape." << std::endl;
            return;
        }
        if (new_dims != 1){
            std::cerr << "Wrong dimension on output array." << std::endl;
            return;
        }
        int rows = *(PyArray_DIMS(ptr));        //Get numpy array shape
        int cols = *(PyArray_DIMS(ptr)+1);
        int mask_rows = *(PyArray_DIMS(mask_ptr));
        int mask_cols = *(PyArray_DIMS(mask_ptr)+1);
        int polar_angles = *(PyArray_DIMS(ccf_ptr));

        if (rows != mask_rows || cols != mask_cols){
            std::cerr << "Arrays have different shape." << std::endl;
            return;
        }

        if (ptr->descr->elsize != sizeof(float) //Test numpy array type
            || mask_ptr->descr->elsize != sizeof(float)
            || ccf_ptr->descr->elsize != sizeof(float))
        {
            std::cerr << "Must be numpy.float32 ndarray" << std::endl;
            std::cerr << "CCF type size is " << ccf_ptr->descr->elsize << std::endl;
            return;
        }

        float **finput = new float*[rows];
        float **fmask = new float*[rows];
        float *foutput;
        for ( int n=0; n<rows; n++ ){
            finput[n] = static_cast<float*> PyArray_GETPTR2(ptr, n, 0);
            fmask[n] = static_cast<float*> PyArray_GETPTR2(mask_ptr, n, 0);
        }
        foutput = static_cast<float*> PyArray_GETPTR1(ccf_ptr,0);

        CudaReprojectAndCorrelate(finput, fmask, foutput, 
                                  rows, cols, r_min, r_max, polar_angles,
                                  center_y, center_x, cval);

        delete[] finput;
        delete[] fmask;

        return;
    }
}

BOOST_PYTHON_MODULE(libcorr)
{
    numeric::array::set_module_and_type( "numpy", "ndarray");
    // Add regular functions to the module.
    def("CorrelateLine", CorrealateLine);
    def("ReprojectToPolar", ReprojectToPolar);
    def("ReprojectAndCorrealate", ReprojectAndCorrealate);
}