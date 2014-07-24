#! /usr/bin/python

import numpy as np
import scipy as sp
import sys
import re
import proj
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import libcorr

BIN_SIZE=4
IMG_SIZE_X=1025
IMG_SIZE_Y=1045
        

def Correlate2d(data):
    # return np.array([sp.correlate(p, np.concatenate((p,p))) for p in data])
    # return np.array([np.sum((sp.correlate(q, np.concatenate((p,p))) for p in data),axis=0) for n,q in enumerate(data)])
    s = []
    for n,q in enumerate(data):
        c_tmp = []
        for p in data:
            c = sp.correlate(q, np.concatenate((p,p)))
            c_tmp.append(c[:-1])
        s.append(np.sum(c_tmp,axis=0))

    return np.array(s)

def main():
    lim = 26
    input = None
    for i in range(25,lim):
        sys.stderr.write('%d\n' % i)
        image_data = np.fromfile('../shuffeled_2/%.04d.raw' % i, dtype=np.float32).reshape((IMG_SIZE_Y, IMG_SIZE_X))

        origin = (IMG_SIZE_X//2,IMG_SIZE_Y//2)
        polar_data = proj.reproject_image_into_polar( image_data, origin )
        
        limit_edge = proj.GetLimitEdge(image_data)
        limit_edge_inner,limit_edge_outer = proj.GetEdgesForPolarByIntensity(polar_data,limit_edge)

        considred_data = polar_data[limit_edge_inner:limit_edge_outer]
        ccf_2d_np = Correlate2d( considred_data )
        size_y,size_x = considred_data.shape
        ccf_2d_c_bind = np.zeros(size_y*size_x,dtype=np.float32).reshape((size_y, size_x))

        libcorr.Correlate(considred_data,ccf_2d_c_bind)

        print considred_data
        print ccf_2d_np
        print ccf_2d_c_bind
        
        img = plt.imshow( ccf_2d_c_bind )
        plt.colorbar()
        plt.pause(10)
        plt.draw()

        img = plt.imshow( ccf_2d_np )
        plt.colorbar()
        plt.pause(10)
        plt.draw()

if __name__ == '__main__':
    main()
