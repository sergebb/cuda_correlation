#! /usr/bin/python

import sys
import time
import numpy as np
import scipy as sp
import proj
import matplotlib.pyplot as plt
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
    return np.array([np.sum((sp.correlate(q, np.concatenate((p, p[:-1]))) for p in data),axis=0) for n,q in enumerate(data)])
    
def RecoverGap(polar_data,polar_mask):
    size_y, size_x = polar_data.shape
    recovered_data = polar_data.copy()
    for y in range(size_y):
        sum = 0.0
        non_mask = 0
        for x in range(size_x):
            val = polar_data[y, x]
            mask_val = polar_mask[y, x]
            if val < 0:
                val = 0
            if mask_val >= 0:
                sum += val
                non_mask += 1
        sum /= non_mask
        for x in range(size_x):
            mask_val = polar_mask[y, x]
            if mask_val < 0:
                recovered_data[y, x] = sum
    return recovered_data


def main():
    lim = 26
    input = None
    for i in range(25, lim):
        sys.stderr.write('%d\n' % i)
        image_data = np.fromfile('../shuffeled_2/%.04d.raw' % i, \
                                dtype=np.float32).reshape((IMG_SIZE_Y, IMG_SIZE_X))

        mask = np.zeros_like(image_data)
        mask[:,500:560] = -10000
        size_y, size_x = image_data.shape

        origin = (IMG_SIZE_X/2.0, IMG_SIZE_Y/2.0)
        with Timer() as t:
            polar_data = proj.reproject_image_into_polar(image_data, origin)
            polar_mask = proj.reproject_image_into_polar(mask, origin)

            limit_edge_inner = 50 + 10
            limit_edge_outer = min(size_x//4, size_y//4)

            considred_data = polar_data[limit_edge_inner:limit_edge_outer]
            considred_mask = polar_mask[limit_edge_inner:limit_edge_outer]
            ccf_2d_np = proj.correlate_by_angle(considred_data, considred_mask)
            ccf_a = proj.combine_ccf_for_angles(ccf_2d_np)

        print('Numpy calculation took %.03f sec.' % t.interval)


        # cuda_polar = np.zeros_like(considred_data)
        ccf_line_c_bind = np.zeros_like(ccf_a,dtype=np.float32)

        with Timer() as t:
            ccf_data = np.zeros_like(considred_data)
            # libcorr.ReprojectToPolar(image_data, cuda_polar, origin[1], origin[0], limit_edge_inner, limit_edge_outer, 0)
            # libcorr.ReprojectToPolar(mask, considred_mask, origin[1], origin[0], limit_edge_inner, limit_edge_outer, 0)
            # libcorr.CorrelateLine(considred_data, considred_mask, ccf_line_c_bind)
            libcorr.ReprojectAndCorrealate(image_data, mask, ccf_line_c_bind, origin[1], origin[0], limit_edge_inner, limit_edge_outer, 0)

        print('Cuda calculation took %.03f sec.' % t.interval)

        fig = plt.figure('Comparison')

        # show first image
        ax1 = fig.add_subplot(1, 3, 1)
        plt.plot(ccf_a)
        # plt.colorbar()

        # show the second image
        fig.add_subplot(1, 3, 2, sharex=ax1)
        plt.plot(ccf_line_c_bind)
        # plt.colorbar()

        ax2 = fig.add_subplot(1, 3, 3, sharex=ax1)
        plt.plot(ccf_line_c_bind-ccf_a)
        # plt.colorbar()

        # show the images
        plt.show()

        # limit_edge_inner = 50 + 10
        # limit_edge_outer = min(size_x//4, size_y//4)

        # considred_data = polar_data[limit_edge_inner:limit_edge_outer]
        # considred_mask = np.zeros_like(considred_data)
        # considred_mask[:,230:260] = -10000
        # with Timer() as t:
        #     ccf_2d_np = proj.correlate_by_angle(considred_data, considred_mask)
        # print('Numpy correlation call took %.03f sec.' % t.interval)
        # ccf_line_c_bind = np.zeros_like(considred_data)

        # recovered_data = RecoverGap(considred_data,considred_mask)

        # with Timer() as t:
        #     libcorr.CorrelateFull(considred_data,ccf_2d_c_bind)

        # print('Cuda function call took %.03f sec.' % t.interval)

        # with Timer() as t:
        #     libcorr.CorrelateLine(considred_data, considred_mask, ccf_line_c_bind)

        # print('Cuda function for line correlation call took %.03f sec.' % t.interval)
        # # print considred_data
        # print ccf_2d_np[0,:20]
        # print ccf_line_c_bind[0,:20]

        # img = plt.imshow( ccf_2d_np - ccf_line_c_bind )
        # plt.colorbar()
        # plt.show()

        # img = plt.imshow(ccf_2d_c_bind)
        # plt.colorbar()
        # plt.pause(10)
        # plt.draw()

        # img = plt.imshow( ccf_2d_np )
        # plt.colorbar()
        # plt.pause(10)
        # plt.draw()

if __name__ == '__main__':
    main()
