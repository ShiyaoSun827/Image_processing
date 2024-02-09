import numpy as np 
from skimage import io, img_as_ubyte
import matplotlib.pyplot as plt
import math

def part1():
    """add your code here"""
    '''
    Read the grayscale image moon.png. 
    Write a function for filtering (moving window dot product) 
    which receives an image and a kernel/filter (of arbitrary size) 
    and outputs the filtered image which has the same size as the input image.
    Using your filtering function, filter the grayscale image with the following filters 
    and display the results.
    Write your own code to implement the following filters: 
    (You cannot use any built-in functions from any library for this.)
    '''
    filename1 = 'moon.png'
    #read in image,"MOON"
    source_gs = io.imread(filename1,
                     as_gray = True

    )
    source_gs = img_as_ubyte(source_gs)

    #Lap filter
    Lap = filtering(source_gs,'Lap')
    pass
def filtering(source_gs,filter):
    '''
    parameters:
                source_gs: img
                filter: kernel/filter
    outputs:
                filtered_img:filtered image

    '''
    if filter == 'Lap':
        kernel = np.array([[0,1,0],
                           [1,4,1],
                           [0,1,0]

        ])
        #padding first,get the h value and w value
        h_kernel = kernel.shape[0]
        w_kernel = kernel.shape[1]
        h_padding = np.cast['int']((h_kernel - 1.) / 2.)
        w_padding = np.cast['int']((w_kernel - 1.) / 2.)
        pad_img = np.zeros((source_gs.shape[0] + h_kernel*2,source_gs.shape[1] + w_kernel*2))
        H = pad_img.shape[0]
        W = pad_img.shape[1]
        #get the height and width of the source and the kernel
        h = source_gs.shape[0]
        w = source_gs.shape[1]
        #assign the gray-scale to the pad_img
        pad_img[h_kernel:(h_kernel + h),w_kernel:(w_kernel+w)] = source_gs
        #for i from H-h-1,for j from W-w-1,for m from -h to h,for n from -w to w
        output = np.zeros((h+h_kernel*2,w+w_kernel*2))
        for i in np.arange(H,H-h_kernel,1):
            for j in np.arange(W,W-w_kernel,1):
                for m in np.arange(-h_kernel,h_kernel+1,1):
                    for n in np.arange(-w_kernel,w_kernel+1,1):
                        output[i,j] += source_gs[i+m,j+n] * kernel[m+h_kernel,n+w_kernel]







def part2():
    """add your code here"""


def part3():
    """add your code here"""


def part4():
    """add your code here"""


def part5():
    """add your code here"""

if __name__ == '__main__':
    part1()
    part2()
    part3()
    part4()
    part5()



