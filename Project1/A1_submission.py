"""Include your imports here
Some example imports are below"""

import numpy as np 
from skimage import io, img_as_ubyte
import matplotlib.pyplot as plt
import math


def part1_histogram_compute():
    filename = r'test.jpg'
    image = io.imread(filename, as_gray=True)
    img = img_as_ubyte(image)


    """add your code here"""



    n = 64
    h = img.shape[0]
    w = img.shape[1]
    hist = np.zeros(n)
    for i in np.arange(0,h,1):
        for j in np.arange(0,w,1):
            temp = img[i,j] // (256/64)
            hist[int(temp)] += 1

    

    hist_np, _ = np.histogram(img,bins = 64,range=[0,256]) # Histogram computed by numpy


    """Plotting code provided here
    Make sure to match the variable names in your code!"""

    plt.figure(figsize=(8, 10))
    plt.subplot(121), plt.plot(hist), plt.title('My Histogram')
    plt.xlim([0, n])
    plt.subplot(122), plt.plot(hist_np), plt.title('Numpy Histogram')
    plt.xlim([0, n])

    plt.show()


def part2_histogram_equalization():
    filename = r'test.jpg'
    image = io.imread(filename, as_gray=True)
    img = img_as_ubyte(image)


    
    """add your code here"""
  
    n_bins = 64
    h = img.shape[0]
    w = img.shape[1]
    hist = np.zeros(n_bins)
    for i in np.arange(0,h,1):
        for j in np.arange(0,w,1):
            temp = img[i,j] // (256/64)
            hist[int(temp)] += 1

    cum_hist = np.zeros(n_bins)
    cum_hist[0] = hist[0]
    for t in np.arange(1,n_bins,1):
        cum_hist[t] = cum_hist[t-1] + hist[t]
    img_eq1 = np.zeros((h,w))    
    for i in np.arange(0,h,1): 
        for j in np.arange(0,w,1):
            img_eq1[i,j] = np.floor((255.0/(h*w))*cum_hist[img[i,j] // 4]+0.5)
    hist_eq = np.zeros(64) 
    for i in np.arange(0,h,1): 
        for j in np.arange(0,w,1): 
            num = img_eq1[i,j] // 4  
            hist_eq[int(num)] += 1 

    # 64-bin Histogram computed by your code (cannot use in-built functions!)
    #hist = ...

    ## HINT: Initialize another image (you can use np.zeros) and update the pixel intensities in every location

    #img_eq1 = ...# Equalized image computed by your code
    

    # Histogram of equalized image
    #hist_eq = ...

    """Plotting code provided here
    Make sure to match the variable names in your code!"""

    plt.figure(figsize=(8, 10))
    plt.subplot(221), plt.imshow(image, 'gray'), plt.title('Original Image')
    plt.subplot(222), plt.plot(hist), plt.title('Histogram')
    plt.xlim([0, n_bins])
    plt.subplot(223), plt.imshow(img_eq1, 'gray'), plt.title('New Image')
    plt.subplot(224), plt.plot(hist_eq), plt.title('Histogram After Equalization')
    plt.xlim([0, n_bins])
    
    plt.show()   


def part3_histogram_comparing():

    filename1 = 'day.jpg'
    filename2 = 'night.jpg'

    # Read in the image
    img1 = io.imread(filename1, as_gray=True)
    # Read in another image
    img2 = io.imread(filename2, as_gray=True)
    
    """add your code here"""

    # Calculate the histograms for img1 and img2 (you can use skimage or numpy)
    hist1, _ = ...
    hist2, _ = ...

    # Normalize the histograms for img1 and img2
    hist1_norm = ...
    hist2_norm = ...

    # Calculate the Bhattacharya coefficient (check the wikipedia page linked on eclass for formula)
    # Value must be close to 0.87
    bc = ...

    print("Bhattacharyya Coefficient: ", bc)


def part4_histogram_matching():
    filename1 = 'day.jpg'
    filename2 = 'night.jpg'

    #============Grayscale============

    # Read in the image
    source_gs = io.imread(filename1,
                           as_gray=True
                           )
    source_gs = img_as_ubyte(source_gs)
    # Read in another image
    template_gs = io.imread(filename2,
                             as_gray=True
                             )
    template_gs = img_as_ubyte(template_gs)
    
    
    """add your code here"""
    matched_gs = ...

    fig = plt.figure()
    gs = plt.GridSpec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)

    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(source_gs, cmap=plt.cm.gray)
    ax1.set_title('source_gs')
    ax2.imshow(template_gs, cmap=plt.cm.gray)
    ax2.set_title('template_gs')
    ax3.imshow(matched_gs, cmap=plt.cm.gray)
    ax3.set_title('matched_gs')
    plt.show()


    #============RGB============
    # Read in the image
    source_rgb = io.imread(filename1,
                           # as_gray=True
                           )
    # Read in another image
    template_rgb = io.imread(filename2,
                             # as_gray=True
                             )
    

    """add your code here"""
    ## HINT: Repeat what you did for grayscale for each channel of the RGB image.
    matched_rgb = ...
    
    fig = plt.figure()
    gs = plt.GridSpec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)

    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(source_rgb)
    ax1.set_title('source_rgb')
    ax2.imshow(template_rgb)
    ax2.set_title('template_rgb')
    ax3.imshow(matched_rgb)
    ax3.set_title('matched_rgb')
    plt.show()

if __name__ == '__main__':
    part1_histogram_compute()
    part2_histogram_equalization()
    part3_histogram_comparing()
    part4_histogram_matching()
