#! /usr/bin/python

import numpy as np
import scipy as sp
import sys
import re
import time
import proj
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import libcorr

BIN_SIZE=4
IMG_SIZE_X=1025
IMG_SIZE_Y=1045
        
class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

def Correlate2d(data):
    # return np.array([sp.correlate(p, np.concatenate((p,p))) for p in data])
    return np.array([np.sum((sp.correlate(q, np.concatenate((p,p[:-1]))) for p in data),axis=0) for n,q in enumerate(data)])
    # s = []
    # for n,q in enumerate(data):
    #     c_tmp = []
    #     for p in data:
    #         c = sp.correlate(q, np.concatenate((p,p)))
    #         c_tmp.append(c[:-1])
    #     s.append(np.sum(c_tmp,axis=0))

    # return np.array(s)

def correlate_by_angle(data):
    arr = np.zeros(data.shape)
    for l,p in enumerate(data):
        for i in range(len(p)):
            p2 = np.roll(p,i)
            c = sp.correlate(p, p2)
            n = np.count_nonzero(np.multiply(p,p2))

            if n is 0:
                # print c,n
                n = 1

            arr[l,i] = c/float(n)

        # c = sp.correlate(p, np.concatenate((p, p)))/float(np.count_nonzero(p))

    return arr

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
        with Timer() as t:
            ccf_2d_np = correlate_by_angle( considred_data )
        print('Numpy correlation call took %.03f sec.' % t.interval)
        # ccf_2d_c_bind = np.zeros_like(considred_data)
        ccf_line_c_bind = np.zeros_like(considred_data)

        # with Timer() as t:
        #     libcorr.CorrelateFull(considred_data,ccf_2d_c_bind)

        # print('Cuda function call took %.03f sec.' % t.interval)

        with Timer() as t:
            libcorr.CorrelateLine(considred_data,ccf_line_c_bind)

        print('Cuda function for line correlation call took %.03f sec.' % t.interval)
        # print considred_data
        print ccf_2d_np[0,:]
        print ccf_line_c_bind[0,:]

        img = plt.imshow( ccf_line_c_bind )
        plt.colorbar()
        plt.pause(10)
        plt.draw()
        
        # img = plt.imshow( ccf_2d_c_bind )
        # plt.colorbar()
        # plt.pause(10)
        # plt.draw()

        # img = plt.imshow( ccf_2d_np )
        # plt.colorbar()
        # plt.pause(10)
        # plt.draw()

if __name__ == '__main__':
    main()
