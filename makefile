
all: libcorr.so

libcorr.so: corr.o cuda_compute.o
	g++ -shared -Wl,-soname,libcorr.so -o libcorr.so corr.o cuda_compute.o -lpython2.7 -lboost_python -lcudart -L/usr/local/cuda/lib64/

corr.o: corr.c cuda_compute.h
	g++ -c -fPIC corr.c -o corr.o -I/usr/include/python2.7/

cuda_compute.o: cuda_compute.cu cuda_compute.h
	nvcc cuda_compute.cu -c -Xcompiler -fpic -arch="sm_20"