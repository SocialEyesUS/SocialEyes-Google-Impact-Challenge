"""
Microaneurysm detector for retinal images
Author: Alex Izvorski, July-August 2015
"""

import numpy
import math
import skimage.filters
import scipy.ndimage
from numba import jit

"""
Create a circular kernel convolved with a gaussian
A positive-negative offset pair of these is used in all the rest of the transforms
"""
def make_kernel(size=17, k_r=6, sigma=1):
    kernel = numpy.zeros((size,size))
    for y in range(kernel.shape[0]):
        for x in range(kernel.shape[1]):
            d = math.sqrt((x-size//2)**2 + (y-size//2)**2)
            if d < k_r:
                kernel[y,x] = 1
    kernel = skimage.filters.gaussian_filter(kernel, sigma)
    return kernel

"""
Transform image into a measure of the dot-like-ness of each area
Dot-like-ness is roughly how much this area differs from its most similar neighbor
"""
@jit
def dot_transform(img, r):
    M, N = img.shape
    output = numpy.full( (M,N), (-1e+6) )
    
    for angle in range(0,360,15):
        di1, dj1 = int(r*math.sin(math.radians(angle))), int(r*math.cos(math.radians(angle)))
        for i in range(M):
            for j in range(N):
                i1, j1 = i + di1, j + dj1
                if i1 < 0 or i1 >= M or j1 < 0 or j1 >= N:
                    continue
                output[i,j] = max(output[i,j], img[i,j] - img[i1,j1])
    return output

"""
Transform image into a measure of the edge-like-ness of each area
Edge-like-ness is roughly how much this area's least similar neighbors differ from each other, in a direction tangential to the direction to the area
Because of the construction, areas immediately adjacent to an edge are still not considered edge-like; the edge has to pass through
"""
@jit
def spd_transform(img, r, angle_step):
    M, N = img.shape
    output = numpy.zeros( img.shape )
    
    for angle in range(0,360,15):
        di1, dj1 = int(r*math.sin(math.radians(angle))), int(r*math.cos(math.radians(angle)))
        di2, dj2 = int(r*math.sin(math.radians(angle+angle_step))), int(r*math.cos(math.radians(angle+angle_step)))
        for i in range(M):
            for j in range(N):
                i1, j1 = i + di1, j + dj1
                i2, j2 = i + di2, j + dj2
                if i1 < 0 or i1 >= M or j1 < 0 or j1 >= N:
                    continue
                if i2 < 0 or i2 >= M or j2 < 0 or j2 >= N:
                    continue
                output[i,j] = max(output[i,j], abs(img[i1,j1] - img[i2,j2]))
    return output

"""
Calculate a scaling value for the other two transforms
This makes them independent of the range of values in the image
"""
@jit
def scale_value(img, r):
    M, N = img.shape
    output = numpy.zeros( (M,N,8) )
    
    for k in range(8):
        angle = k*45
        di1, dj1 = int(r*math.sin(math.radians(angle))), int(r*math.cos(math.radians(angle)))
        for i in range(M):
            for j in range(N):
                i1, j1 = i + di1, j + dj1
                if i1 < 0 or i1 >= M or j1 < 0 or j1 >= N:
                    continue
                output[i,j,k] = img[i,j] - img[i1,j1]
    return numpy.std(output)


"""
Find microaneurysms
Produces a mask that identifies the potential regions, with same dimensions as the source image
"""
def ma_detect(img, kernel=None, s_threshold=0.3, d_threshold=-0.07, s_r=12, d_r=12, angle_step=30):
    if kernel == None:
        kernel = make_kernel()

    img_filt = scipy.ndimage.convolve(img, kernel)

    d = dot_transform(img_filt, d_r)
    s = spd_transform(img_filt, s_r, angle_step)

    scale = scale_value(img_filt, d_r)
    d = d / scale
    s = s / scale

    ma_image = (s < s_threshold) * (d < d_threshold)
    return ma_image
