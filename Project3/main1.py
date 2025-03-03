# import statements
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage import io

def part1():
    """ BasicBayer: reconstruct RGB image using GRGB pattern"""
    filename_Grayimage = 'PeppersBayerGray.bmp'
    filename_gridB = 'gridB.bmp'
    filename_gridR = 'gridR.bmp'
    filename_gridG = 'gridG.bmp'

    # read image
    img = io.imread(filename_Grayimage, as_gray =True)
    h,w = img.shape

    # our final image will be a 3 dimentional image with 3 channels
    rgb = np.zeros((h,w,3),np.uint8)

    # reconstruction of the green channel IG
    IG = np.copy(img) # copy the image into each channel

    for row in range(0,h,4): # loop step is 4 since our mask size is 4.
        for col in range(0,w,4): # loop step is 4 since our mask size is 4.
            # TODO: compute pixel value for each location where mask is unshaded (0) 
            # interpolate each pixel using its every valid (shaded) neighbour
            #IG[row,col+1]= (int(img[row,col])+int(img[row,col+2])+int(img[row+1,col+1]))/3  # B (recommendation: add this kinf of inline comments to each line within for loop)
            # ...
            #IG[row+3,col]= (int(img[row+2,col])+int(img[row+3,col+1]))/2                    # M
            # ...
            
            #B = (A+G+F)/3
            IG[row,col+1]= (int(img[row,col])+int(img[row,col+2])+int(img[row+1,col+1]))/3
            #D = (C+H)/2
            IG[row,col+3]= (int(img[row,col+2])+int(img[row+1,col+3]))/2
            #E = (A +F+I)/3
            IG[row+1,col]= (int(img[row,col])+int(img[row+1,col+1])+int(img[row+2,col]))/3
            #G = (F+C++k+H)/4
            IG[row+1,col+2]= (int(img[row+1, col+1])+int(img[row, col+2])+int(img[row+2, col+2])+int(img[row+1, col+3]))/4
            #J = （F+I+K+N）/4
            IG[row+2,col+1]= (int(img[row+1, col+1])+int(img[row+2, col+2])+int(img[row+2, col])+int(img[row+3, col+1]))/4
            #L = (H+K+P)/3
            IG[row+2,col+3]= (int(img[row+1,col+3])+int(img[row+2,col+2])+int(img[row+3,col+3]))/3
            #M = （I+N）/2
            IG[row+3,col]= (int(img[row+2,col])+int(img[row+3,col+1]))/2
            #O = （N+K+P）/3
            IG[row+3,col+2]= (int(img[row+3,col+1])+int(img[row+2,col+2])+int(img[row+3,col+3]))/3


         

    # TODO: show green (IR) in first subplot (221) and add title - refer to rgb one for hint on plotting
    plt.figure(figsize=(10,8))
    # ...
    img = io.imread(filename_Grayimage, as_gray =True)

    IR = np.copy(img)

    for row in range(0,h,4):
        for col in range(0,w,4): 
            
            IR[row, col+2] = (int(IR[row, col+1]) + int(IR[row, col+3]))/2
            IR[row+1, col+1] = (int(IR[row, col+1]) + int(IR[row+2, col+1]))/2
            IR[row+1, col+3] = (int(IR[row, col+3]) + int(IR[row+2, col+3]))/2
            IR[row+2, col+2] = (int(IR[row+2, col+1]) + int(IR[row+2, col+3]))/2
            IR[row+1, col+2] = (int(IR[row, col+1]) + int(IR[row, col+3])+ int(IR[row+2, col+1])+int(IR[row+2, col+3]))/4

            IR[row, col] = int(IR[row, col+1]) 
            IR[row+1, col] = int(IR[row+1, col+1]) 
            IR[row+2, col] = int(IR[row+2, col+1]) 
            IR[row+3, col+1] = int(IR[row+2, col+1]) 
            IR[row+3, col] = int(IR[row+3, col+1]) 
            IR[row+3, col+2] = int(IR[row+2, col+2]) 
            IR[row+3, col+3] = int(IR[row+2, col+3])



    # TODO: reconstruction of the red channel IR (simiar to loops above), 
    # 
    # TODO: show IR in second subplot (224) and title
    # ...

    # TODO: reconstruction of the blue channel IB (similar to loops above), 
    # ...
    # TODO: show IB in third subplot () and title
    # ...
    img = io.imread(filename_Grayimage, as_gray =True)
    IB = np.copy(img)

    for row in range(0,h,4):
        for col in range(0,w,4): 


            IB[row+1, col+1] = (int(IB[row+1, col]) + int(IB[row+1, col+2]))/2
            IB[row+2, col] = (int(IB[row+1, col]) + int(IB[row+3, col]))/2
            IB[row+2, col+1] = (int(IB[row+1, col]) + int(IB[row+1, col+2]) + int(IB[row+3, col]) + int(IB[row+3, col+2]))/4
            IB[row+2, col+2] = (int(IB[row+1, col+2]) + int(IB[row+3, col+2]))/2
            IB[row+3, col+1] = (int(IB[row+3, col]) + int(IB[row+3, col+2]))/2
            
            IB[row, col] = int(IB[row+1, col])
            IB[row, col+1] = int(IB[row+1, col+1])
            IB[row, col+2] = int(IB[row+1, col+2])
            IB[row, col+3] = int(IB[row+1, col+2])
            IB[row+1, col+3] = int(IB[row+1, col+2])
            IB[row+2, col+3] = int(IB[row+2, col+2])
            IB[row+3, col+3] = int(IB[row+3, col+2])





    # merge the channels
    rgb[:, :, 0] = IR
    rgb[:, :, 1] = IG
    rgb[:, :, 2] = IB



    # TODO: merge the three channels IG, IB, IR in the correct order
    #rgb[:,:,1]=IG
    # ...


    # TODO: show rgb image in final subplot (224) and add title
    plt.subplot(221)
    plt.imshow(IG,cmap='gray')
    plt.title('IG')
    
    plt.subplot(222)
    plt.imshow(IR,cmap='gray')
    plt.title('IR')
    
    plt.subplot(223)
    plt.imshow(IB,cmap='gray')
    plt.title('IB')
   
    plt.subplot(224)
    plt.imshow(rgb)
    plt.title('rgb')
    plt.show()

if __name__  == "__main__":
    part1()