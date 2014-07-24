#include <Python.h>
#include <string>
#include <boost/python.hpp>
#include <stdio.h>
#include <iostream>
#include <numpy/ndarrayobject.h> // ensure you include this header
#include "cuda_compute.h"
using namespace boost::python;

namespace { // Avoid cluttering the global namespace.

    // A couple of simple C++ functions that we want to expose to Python.
    std::string greet() { return "hello, world"; }
    int square(int number) { return number * number; }

    void CorrealateLine(float* data,float* newdata,int size){
        float sum;
        int i,j,p;
        for ( i=0; i<size; i++ ){
            sum=0;
            for ( j=0; j<size; j++ ){
                p = (i+j)%size;
                sum+=data[j]*data[p];
            }
            newdata[i]=sum;
        }
    }

    void CorrealateArray(float** input_data,float** output_data, int rows, int cols){
        float *line_1,*line_2;
        float sum_tmp;
        int i,j,k,l,p;
        float *tmp_line = new float[cols];
        float *sum_line = new float[cols];
        for( i = 0; i < rows; i++ ){
            std::fill( sum_line, sum_line + cols, 0 );
            for( j = 0; j < rows; j++ ){
                line_1 = input_data[i];
                line_2 = input_data[j];
                for( k = 0; k < cols; k++ ){
                    sum_tmp = 0;
                    for ( l = 0; l < cols; l++ ){           //Compute correlation between 2 rows
                        p = (k+l)%cols;
                        sum_tmp += line_1[p]*line_2[l];
                    }                                       
                    tmp_line[k] = sum_tmp;
                }
                for( k = 0; k < cols; k++ ){
                    sum_line[k] += tmp_line[k];             //Sum correlation for same line_1
                }
            }
            for( k = 0; k < cols; k++ ){
                output_data[i][k] = sum_line[k];
            }
        }
        delete[] tmp_line;
    }

    void Correalate( boost::python::numeric::array& data, boost::python::numeric::array& newdata )
    {

        PyArrayObject *ptr      = (PyArrayObject *) data.ptr();   //Get numpy data ptr
        PyArrayObject *new_ptr  = (PyArrayObject *) newdata.ptr();
        if (ptr == NULL) {
            std::cerr << "Could not get NP array." << std::endl;
            return;
        }
        if (new_ptr == NULL) {
            std::cerr << "Could not get NP array." << std::endl;
            return;
        }
        const int dims     =  PyArray_NDIM(ptr);    //Get numpy array dimension
        const int new_dims =  PyArray_NDIM(new_ptr);    //Get numpy array dimension
        if (dims != 2){
            std::cerr << "Wrong dimension on array." << std::endl;
            return;
        }
        if (dims != new_dims){
            std::cerr << "Wrong dimension on new array." << std::endl;
            return;
        }
        int rows = *(PyArray_DIMS(ptr));        //Get numpy array size
        int cols = *(PyArray_DIMS(ptr)+1);
        int new_rows = *(PyArray_DIMS(new_ptr));        //Get numpy array size
        int new_cols = *(PyArray_DIMS(new_ptr)+1);

        if (rows != new_rows || cols != new_cols ){
            std::cerr << "Wrong shape of the new array." << std::endl;
            return;
        }

        if (ptr->descr->elsize != sizeof(float) || new_ptr->descr->elsize != sizeof(float)){   //Test numpy array type
            std::cerr << "Must be numpy.float32 ndarray" << std::endl;
            return;
        }

        float **finput = new float*[rows];
        float **foutput = new float*[rows];
        for ( int n=0; n<rows; n++ ){
            finput[n] = static_cast<float*> PyArray_GETPTR2(ptr, n, 0);
            foutput[n] = static_cast<float*> PyArray_GETPTR2(new_ptr, n, 0);
        }
        // CorrealateArray(finput,foutput,rows,cols);
        CudaCorrelate(finput,foutput,rows,cols);
        // float* newline = new float[cols];
        // for ( int n=0; n<rows; n++ ){
        //     float *line = static_cast<float*> PyArray_GETPTR2(ptr, n, 0);
        //     CorrealateLine(line,newline,cols);
        //     for ( int i=0; i<cols; i++ ){
        //         line[i] = newline[i];
        //     }
        // }

        // delete[] newline;
        delete[] finput;
        delete[] foutput;

        return;
        
    }
}


BOOST_PYTHON_MODULE(libcorr)
{
    numeric::array::set_module_and_type( "numpy", "ndarray");
    // Add regular functions to the module.
    def("greet", greet);
    def("square", square);
    def("Correlate", Correalate);
}