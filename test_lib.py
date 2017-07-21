#!/usr/bin/env python

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
    image_data = []
    for i in range(1, 300):
        sys.stderr.write('%d\n' % i)
        image = np.fromfile('../shuffeled_2/%.04d.raw' % i, \
                                dtype=np.float32).reshape((IMG_SIZE_Y, IMG_SIZE_X))
        image[image < 0] = 0
        # image[500:540, :] = -10000
        image_data.append(image)

    image_data = np.array(image_data, dtype=np.float32)

    num_images, size_y, size_x = image_data.shape

    limit_edge_inner = 0
    limit_edge_outer = min(size_x//4, size_y//4)
    origin = (size_x/2.0, size_y/2.0)

    ccf_data = []
    with Timer() as t:
        for image in image_data:
            mask = np.zeros_like(image)
            mask[image < 0] = -10000

            polar_data = proj.reproject_image_into_polar(image, origin)
            polar_mask = proj.reproject_image_into_polar(mask, origin)

            considred_data = polar_data[limit_edge_inner:limit_edge_outer]
            considred_mask = polar_mask[limit_edge_inner:limit_edge_outer]

            ccf_2d_np = proj.correlate_by_angle(considred_data, considred_mask)
            ccf_a = proj.combine_ccf_for_angles(ccf_2d_np)
            ccf_data.append(ccf_a)

        ccf_data = np.array(ccf_data)

    print('Numpy calculation took %.03f sec.' % t.interval)

    # cpu_ccf_2d_data = np.array(cpu_ccf_2d_data)

    gpu_ccf_data = np.zeros_like(ccf_data, dtype=np.float32)

    with Timer() as t:
        libcorr.ReprojectAndCorrealate(image_data, gpu_ccf_data, origin[1], origin[0], limit_edge_inner, limit_edge_outer, 0)

    print('Cuda calculation took %.03f sec.' % t.interval)

    fig = plt.figure('Comparison')

    # for i in range(num_images):
    #     ccf_cpu = ccf_data[i, :]
    #     ccf_gpu = gpu_ccf_data[i, :]

    #     ccf_cpu /= ccf_cpu[0]
    #     ccf_gpu /= ccf_gpu[0]

    #     fig = plt.figure('Comparison')
    #     ax1 = fig.add_subplot(1, 3, 1)
    #     plt.plot(ccf_cpu)
    #     # plt.imshow(ccf_data)
    #     # plt.colorbar()

    #     ax2 = fig.add_subplot(1, 3, 2, sharex=ax1)
    #     plt.plot(ccf_gpu)
    #     # plt.imshow(gpu_ccf_data)
    #     # plt.colorbar()

    #     ax3 = fig.add_subplot(1, 3, 3, sharex=ax1)
    #     plt.plot(ccf_cpu-ccf_gpu)
    #     # plt.imshow(ccf_data-gpu_ccf_data)
    #     # plt.colorbar()

    #     ax1.set_ylim([0, 1])
    #     ax2.set_ylim([0, 1])
    #     ax3.set_ylim([0, 1])

    #     plt.show()

if __name__ == '__main__':
    main()
