import os
from sklearn.cluster import KMeans
from scipy import spatial
from skimage import io, color, img_as_float
import numpy as np
import matplotlib.pyplot as plt
from math import floor


# Finds the closest colour in the palette using kd-tree.
def nearest(palette, colour):
    dist, i = palette.query(colour)
    return palette.data[i]


# Make a kd-tree palette from the provided list of colours
def makePalette(colours):
    return spatial.KDTree(colours)


# Dynamically calculates an N-colour palette for the given image
# Uses the KMeans clustering algorithm to determine the best colours
# Returns a kd-tree palette with those colours
def findPalette(image, nColours):
    # TODO: perform KMeans clustering to get 'colours' --  the computed k means
    
    a = image.shape
    temp_image = image.copy()
    img_a = temp_image.reshape(-1,a[2])
    kmeans = KMeans(n_clusters = nColours)
    
    kmeans.fit(img_a)
    colours = kmeans.cluster_centers_
    print(colours)
    '''
    
    k = KMeans(n_clusters = nColours).fit(a)
    colours = k.cluster_centers_
    print(colours)
    '''
    return makePalette(colours)


def ModifiedFloydSteinbergDitherColor(image, palette):
    """
    The following pseudo-code for a grayscale image is grabbed from Wikipedia:
    https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering.
    The error distribution has been modified according to the instructions on eClass.

    total_abs_err := 0
    for each y from top to bottom ==> (height)
        for each x from left to right ==> (width)
            oldpixel  := image[x][y]
            newpixel  := nearest(oldpixel) # Determine the new colour for the current pixel from palette
            image[x][y]  := newpixel 
            quant_error  := oldpixel - newpixel

            total_abs_err := total_abs_err + abs(quant_error)

            image[x + 1][y    ] := image[x + 1][y    ] + quant_error * 11 / 26 
            image[x - 1][y + 1] := image[x - 1][y + 1] + quant_error * 5 / 26
            image[x    ][y + 1] := image[x    ][y + 1] + quant_error * 7 / 26
            image[x + 1][y + 1] := image[x + 1][y + 1] + quant_error * 3 / 26

    avg_abs_err := total_abs_err / image.size
    """

    # TODO: implement agorithm for RGB image (hint: you need to handle error in each channel separately)
    h,w,d = image.shape
    for  y in range(h-1):
        for  x in range(w-1):
            oldpixel  = image[x][y]
            newpixel  = nearest(palette,oldpixel) # Determine the new colour for the current pixel from palette
            image[x][y]  = newpixel 
            quant_error  = oldpixel - newpixel

            #total_abs_err = total_abs_err + abs(quant_error)

            image[x + 1][y    ] = image[x + 1][y    ] + quant_error * 11 / 26 
            image[x - 1][y + 1] = image[x - 1][y + 1] + quant_error * 5 / 26
            image[x    ][y + 1] = image[x    ][y + 1] + quant_error * 7 / 26
            image[x + 1][y + 1] = image[x + 1][y + 1] + quant_error * 3 / 26


        
    #avg_abs_err = total_abs_err / image.size

    return image


if __name__ == "__main__":
    # The number colours: change to generate a dynamic palette
    nColours = 7

    # read image
    imfile = 'mandrill.png'
    image = io.imread(imfile)
    orig = image.copy()

    # Strip the alpha channel if it exists
    image = image[:, :, :3]

    # Convert the image from 8bits per channel to floats in each channel for precision
    image = img_as_float(image)

    # Dynamically generate an N colour palette for the given image
    palette = findPalette(image, nColours)
    colours = palette.data
    colours = img_as_float([colours.astype(np.ubyte)])[0]
    

    # call dithering function
    img = ModifiedFloydSteinbergDitherColor(image, palette)

    # show
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(orig), plt.title('Original Image')
    plt.subplot(122), plt.imshow(img), plt.title(f'Dithered Image (nColours = {nColours})')
    plt.show()
