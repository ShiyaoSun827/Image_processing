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
    # 64-bin Histogram computed by your code (cannot use in-built functions!)
    n_bins = 64
    #get height and width of the image
    h = img.shape[0]
    w = img.shape[1]
    
    #Initialize histogram array
    hist = np.zeros(n_bins)

    for i in np.arange(0,h,1):
        for j in np.arange(0,w,1):
            #each bin can represent 4 values
            temp = int(img[i,j] // (256/64))
            hist[temp] += 1

    ## HINT: Initialize another image (you can use np.zeros) and update the pixel intensities in every location
    #initialize cum hist array
    Cumulative_hist = np.zeros(n_bins)
    
    for t in np.arange(0,n_bins,1):
        if t == 0:
            Cumulative_hist[t] = hist[t]
        Cumulative_hist[t] = Cumulative_hist[t-1] + hist[t]
    #equalization:a' = floor[((k-1)/MN)H(a)+0.5]
    img_eq1 = np.zeros((h,w))    
    for i in np.arange(0,h,1): 
        for j in np.arange(0,w,1):
            temp1 = 255/(h*w)
            pos = img[i,j] // 4
            img_eq1[i,j] = np.floor(temp1*Cumulative_hist[pos]+0.5)
    # Histogram of equalized image
    hist_eq = np.zeros(64) 
    for i in np.arange(0,h,1): 
        for j in np.arange(0,w,1): 
            pos = int(img_eq1[i,j] // 4)
            hist_eq[pos] += 1 



    

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
    img1= img_as_ubyte(img1)
    img2 = img_as_ubyte(img2)
    # Calculate the histograms for img1 and img2 (you can use skimage or numpy)
    hist1, _ = np.histogram(img1,bins=256,range=(0,256)) 
    h_day = img1.shape[0]
    w_day = img1.shape[1] 

    hist2, _= np.histogram(img2,bins=256,range=(0,256))
    h_night = img2.shape[0]
    w_night = img2.shape[1]
    
    
    # Normalize the histograms for img1 and img2
    
    # Calculate the Bhattacharya coefficient (check the wikipedia page linked on eclass for formula)
    # Value must be close to 0.87
    
    bc=0 
    #Since normalized version is defined as: H(i)/(MN), MN is the width * height of the image
    for i in np.arange (256): 
        hist1_norm = hist1[i] / (h_day * w_day)
        hist2_norm = hist2[i] /  (h_night * w_night)
        bc += math.sqrt(hist1_norm * hist2_norm) 


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
    #Caculate PA,Cumu hist of input
    hist = np.zeros(n_bins)
    h_source = source_gs.shape[0]
    w_source = source_gs.shape[1]
    for i in np.arange(0,h_source,1):
        for j in np.arange(0,w_source,1):
            #each bin can represent 4 values
            temp = int(source_gs[i,j]) 
            hist[temp] += 1
    n_bins = 256
    Cumulative_hist = np.zeros(n_bins)
    
    for t in np.arange(0,n_bins,1):
        if t == 0:
            Cumulative_hist[t] = hist[t]
        Cumulative_hist[t] = Cumulative_hist[t-1] + hist[t]
    #cacualte the nomalized cum hist
    bc=0 
    #Since normalized version is defined as: H(i)/(MN), MN is the width * height of the image
    for i in np.arange (256): 
        hist1_norm = hist1[i] / (h_day * w_day)
        hist2_norm = hist2[i] /  (h_night * w_night)
        bc += math.sqrt(hist1_norm * hist2_norm) 

    

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
