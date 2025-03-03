import numpy as np 
from skimage import io, img_as_ubyte
import matplotlib.pyplot as plt
import math
import cv2
from skimage import feature
from scipy.spatial import distance
from skimage.filters import laplace

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
    plt.subplot(1, 2, 1) 
    plt.imshow(source_gs, cmap='gray')
    plt.title("Original")
    plt.subplot(1, 2, 2) 
    plt.imshow(Lap, cmap='gray')
    plt.title("Laplace filtered image(1)")
    plt.show()

    #Gaussian filter
    plt.subplot(1, 2, 1) 
    plt.imshow(source_gs, cmap='gray')
    plt.title("Original")
    Gaussian = filtering(source_gs,'Gaussian')
    plt.subplot(1, 2, 2) 
    plt.imshow(Gaussian, cmap='gray')
    plt.title(" Gaussian filtered image(2)")
    plt.show()

    #filter3
    plt.subplot(1, 2, 1) 
    plt.imshow(source_gs, cmap='gray')
    plt.title("Original")
    filterd = filtering(source_gs,'filter1')
    plt.subplot(1, 2, 2) 
    plt.imshow(filterd, cmap='gray')
    plt.title(" filtered image(3) ")
    plt.show()

    #filter 4
    plt.subplot(1, 2, 1) 
    plt.imshow(source_gs, cmap='gray')
    plt.title("Original")
    filterd = filtering(source_gs,'filter2')
    plt.subplot(1, 2, 2) 
    plt.imshow(filterd, cmap='gray')
    plt.title(" filtered image(4) ")
    plt.show()
    
    #Enhanced by Laplace
    plt.subplot(1,2,1)
    plt.imshow((source_gs).astype(int), cmap='gray', vmin=0, vmax=255)
    plt.title("Original")
    plt.subplot(1,2,2)
    out = filtering(source_gs,'enhance1')
    plt.imshow((source_gs+out).astype(int), cmap='gray', vmin=0, vmax=255)
    plt.title("Enhanced by Laplace(5)")
    plt.show()

    #Enhanced by Guassian
    plt.subplot(1,2,1)
    plt.imshow((source_gs).astype(int), cmap='gray', vmin=0, vmax=255)
    plt.title("Original")
    plt.subplot(1,2,2)
    out = filtering(source_gs,'enhance2')
    plt.imshow((source_gs+(source_gs-out)).astype(int), cmap='gray', vmin=0, vmax=255)
    plt.title("Enhanced by Gaussian(6)")
    plt.show()

def filtering(source_gs,filter):
    '''
    parameters:
                source_gs: img
                filter: kernel/filter
    outputs:
                filtered_img:filtered image

    '''
    if filter == 'Lap':
        kernel = np.array([[0,-1,0],
                           [-1,4,-1],
                           [0,-1,0]

        ])
    elif filter =='Gaussian':
        kernel = (1/273) * np.array([
            [1,  4,   7,   4,  1],
            [4, 16, 26, 16, 4],
            [7, 26, 41, 26, 7],
            [4, 16, 26, 16, 4],
            [1,   4,   7,   4, 1]
        ])
        
    elif filter == 'filter1':
        
        kernel = np.array([
            [0,0,0,0,0],
            [0,1,0,1,0],
            [0,0,0,1,0]

        ])
    elif filter == 'filter2':
        kernel = np.array([
            [0,0,0], [6,0,6],[0,0,0]
        ]) 
    elif filter =='enhance1':
        
        
        kernel = np.array([[0,-1,0],
                           [-1,4,-1],
                           [0,-1,0]

        ])
        h = source_gs.shape[0] 
        w = source_gs.shape[1]
        
    
        
        
        h_ker,w_ker = kernel.shape 
    
        h_padding = np.cast['int']((h_ker-1.)/2.) 
        w_padding = np.cast['int']((w_ker-1.)/2.) 
        
        padding = np.zeros((h + h_padding * 2 , w + w_padding *2 )) 
        source_gs = source_gs.astype('float64')
        output_padding = np.zeros_like(source_gs)

        padding[h_padding:(h+h_padding), w_padding:(w+w_padding)] = source_gs

    
        for i in np.arange(h_padding,h-h_padding,1):
            for j in np.arange(w_padding,w-w_padding,1):
                for m in np.arange(-h_padding,h_padding+1,1):
                    for n in np.arange(-w_padding,w_padding+1,1):
                        output_padding[i,j] += source_gs[i+m,j+n]*kernel[m+h_padding,n+w_padding]
                        
        return output_padding
    elif filter == 'enhance2':
        
        kernel = np.array([
            [1,  4,   7,   4,  1],
            [4, 16, 26, 16, 4],
            [7, 26, 41, 26, 7],
            [4, 16, 26, 16, 4],
            [1,   4,   7,   4, 1]
        ])
        kernel = (1/273) * kernel
        h = source_gs.shape[0] 
        w = source_gs.shape[1]
        
     
        h_ker,w_ker = kernel.shape 
    
        h_padding = np.cast['int']((h_ker-1.)/2.) 
        w_padding = np.cast['int']((w_ker-1.)/2.) 
        
        padding = np.zeros((h + h_padding * 2 , w + w_padding *2 )) 
        source_gs = source_gs.astype('float64')
        output_padding = np.zeros_like(source_gs)
        #assign the gray-scale to the pad_img
        #pad_img[h_kernel:(h_kernel + h),w_kernel:(w_kernel+w)] = source_gs

        padding[h_padding:(h+h_padding), w_padding:(w+w_padding)] = source_gs

    #for i from H-h-1,for j from W-w-1,for m from -h to h,for n from -w to w
        for i in np.arange(h_padding,h-h_padding,1):
            for j in np.arange(w_padding,w-w_padding,1):
                for m in np.arange(-h_padding,h_padding+1,1):
                    for n in np.arange(-w_padding,w_padding+1,1):
                        output_padding[i,j] += source_gs[i+m,j+n]*kernel[m+h_padding,n+w_padding]
        return output_padding
    
    

    #padding first,get the h value and w value
    h_kernel = kernel.shape[0]
    w_kernel = kernel.shape[1]
    h_padding = np.cast['int']((h_kernel - 1.) / 2.)
    w_padding = np.cast['int']((w_kernel - 1.) / 2.)
    #pad_img = np.zeros((source_gs.shape[0] + h_kernel*2,source_gs.shape[1] + w_kernel*2))
    
    #pad_img = np.zeros((temp_h + h_padding*2,temp_w + w_padding*2))
    

    pad_img = np.zeros((source_gs.shape[0] + h_padding*2,source_gs.shape[1] + w_padding*2))
    H = pad_img.shape[0]
    W = pad_img.shape[1]
    #get the height and width of the source and the kernel
    h = source_gs.shape[0]
    w = source_gs.shape[1]
    #assign the gray-scale to the pad_img
    #pad_img[h_kernel:(h_kernel + h),w_kernel:(w_kernel+w)] = source_gs
    pad_img[h_padding:(h_padding + h),w_padding:(w_padding+w)] = source_gs
    #for i from H-h-1,for j from W-w-1,for m from -h to h,for n from -w to w
    #output = np.zeros((h+h_kernel*2,w+w_kernel*2))
    output = np.zeros((h+h_padding*2,w+w_padding*2))
    #output = np.zeros_like((source_gs))
    for i in np.arange(h_padding,h-h_padding,1):
        for j in np.arange(w_padding,w-w_padding,1):
            for m in np.arange(-h_padding,h_padding+1,1):
                for n in np.arange(-w_padding,w_padding+1,1):
                    output[i,j] += source_gs[i+m,j+n] * kernel[m+h_padding,n+w_padding]
    return output
    








def part2():
    """add your code here"""
    noisy = io.imread("noisy.jpg") 

    

  
    #apply median filter to remove the noise
    median = cv2.medianBlur(noisy, 5)
    
    #Apply a Gaussian filter to the same noisy image.
    Gau = cv2.GaussianBlur(noisy,(5,5),0) 
 
    
    plt.subplot(1, 3, 1) 
    plt.imshow(noisy, cmap='gray')
    plt.title("Original")
    
    plt.subplot(1, 3, 2) 
    plt.imshow(median, cmap='gray')
    plt.title("median")

    plt.subplot(1, 3, 3) 
    plt.imshow(Gau, cmap='gray')
    plt.title("Gaussian")
    plt.show()
    #Response toward part2: Median is better
    


def part3():
    """add your code here"""
    damage_camera = io.imread("damage_cameraman.png") 
    damage_mask = io.imread("damage_mask.png")
    temp_camera = io.imread("damage_cameraman.png")

    height = damage_camera.shape[0] 
    width =  damage_camera.shape[1]

    damage_pixel = []

    for i in range(height): 
      for j in range(width):
        if damage_mask[i][j].all() == 0: 
          damage_pixel.append((i,j)) 
          # If it is damaged, 
          #add its coords to the damage_pixel.
    
    
    for i in range(3000): 
      #Apply a Gaussian filter to the damaged image 
      Gaussian_image = cv2.GaussianBlur(temp_camera,(3,3),0)
     
      for j in damage_pixel: 
        temp_camera[j] = Gaussian_image[j] 
       
      
            
    plt.subplot(1, 2, 1) 
    plt.imshow(damage_camera, cmap='gray')
    plt.title("damaged image")

    plt.subplot(1, 2, 2) 
    plt.imshow(temp_camera, cmap='gray')
    plt.title("restored image")

    plt.show()



def part4():
    """add your code here"""
    ex2 = io.imread("ex2.jpg")

    height = ex2.shape[0] 
    width =  ex2.shape[1] 

    
    horizontal_kernel = np.array([
        [-1, 0 ,1], 
        [-2 ,0 ,2],
        [-1 ,0 ,1]])
    height_kernel,width_kernel = horizontal_kernel.shape 
   
    #similar to part1
    h_padding = np.cast['int']((height_kernel-1.)/2.) 
    w_padding = np.cast['int']((width_kernel- 1.)/2.)
    

    padding = np.zeros((height + h_padding * 2 , width + w_padding *2 )) 
    output_padding_horizontal = np.zeros((height + h_padding * 2 ,width + w_padding *2))
    #assign the gray-scale to the pad_img
    
    padding[h_padding:(height+h_padding), w_padding:(width+w_padding)] = ex2 
    for i in np.arange(h_padding,height-h_padding,1):    
        for j in np.arange(w_padding,width-w_padding,1):
            for m in np.arange(-h_padding,h_padding+1,1):
                    for n in np.arange(-w_padding,w_padding+1,1):
                        output_padding_horizontal[i,j] += ex2[i+m,j+n]*horizontal_kernel[m+h_padding,n+w_padding]
    
    
    
    #similar to part1
    #set the veritical kernel
    vertical_kernel = np.array([
            [1,  2,  1], 
            [0,  0,  0],
            [-1,-2,-1]]) 
    
    
    padding_veritical = np.zeros((height + h_padding * 2 , width + w_padding *2 )) 
    output_padding_vertical = np.zeros((height + h_padding * 2 ,width + w_padding *2))
    #assign the gray-scale to the pad_img
    
    padding_veritical[h_padding:(height+h_padding), w_padding:(width+w_padding)] = ex2 

    for i in np.arange(h_padding,height-h_padding,1):   
        for j in np.arange(w_padding,width-w_padding,1):
            for m in np.arange(-h_padding,h_padding+1,1):
                    for n in np.arange(-w_padding,w_padding+1,1):
                        output_padding_vertical[i,j] += ex2[i+m,j+n]*vertical_kernel[m+h_padding,n+w_padding]
    
    #combine the horizontal and the veritcal by using sqrt
    E = np.sqrt(output_padding_horizontal**2 + output_padding_vertical**2)
    #Plots the images
    plt.figure(figsize=(9,6))
    plt.subplot(2, 2, 1) 
    plt.imshow(ex2, cmap='gray')
    plt.title("image")
    plt.axis('off')

    plt.subplot(2, 2, 2) 
    plt.imshow(output_padding_horizontal, cmap='gray')
    plt.title("Horizontal")
    plt.axis('off')

    plt.subplot(2, 2, 3 ) 
    plt.imshow(output_padding_vertical, cmap='gray')
    plt.title("Vertical")
    plt.axis('off')

    plt.subplot(2, 2, 4 ) 
    plt.imshow(E, cmap='gray')
    plt.title("Gradient")
    plt.axis('off')

    plt.show()


def part5():
    """add your code here"""
    picture = io.imread("ex2.jpg", as_gray=True) 
    target = io.imread("canny_target.jpg", as_gray=True) 
    
     
    #set values
    low_thresholds = [50,70,90]
    high_thresholds = [150,170,190]
    sigmas = [1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8]
    #best parameters
    #Initialize best_distance to a large number, 
    #such as 1e10, and initialize the best_params array with zeros.
    best_distance = 1e10
    best_params = [0,0,0]

    for low_threshold in low_thresholds:
        for high_threshold in high_thresholds:
            for sigma in sigmas:
                #canny_output = Apply the Canny method with the parameters to the image
                canny = feature.canny(image=picture, low_threshold=low_threshold, high_threshold=high_threshold, sigma=sigma) 
                #this_dist = Compute cosine distance between canny_output image and the target image
                dist = distance.cosine(canny.flatten(),target.flatten())
                #if (this_dist < best_distance) and (np.sum(canny_output>0.0)>0.0), 
                if dist < best_distance:
                    #best_distance = this_dist
                    #Store current parameter values in best_params array
                    best_distance = dist
                    best_params = [low_threshold,high_threshold,sigma]
    #my_image = Apply the Canny method to the image with parameters stored in the  best_params array
    image = feature.canny(image = picture, low_threshold=best_params[0],high_threshold=best_params[1],sigma=best_params[2])
    
    print("The best distance is:",best_distance)
    print(f"The best 'low threshold' = {best_params[0]};\nThe best 'high_threshold' = {best_params[1]};\nThe best 'sigma' = {best_params[2]}")

    #apply a gussian filter to deal with the image
    Gau = cv2.GaussianBlur(picture,(5,5),0) 
    
      
    plt.figure(figsize=(9, 6))         
    plt.subplot(2, 2, 1) 
    plt.imshow(picture, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2, 2, 2) 
    plt.imshow(Gau , cmap='gray')
    plt.title("Guassian")
    plt.axis('off')


    plt.subplot(2, 2, 3) 
    plt.imshow(target, cmap='gray')
    plt.title("Target")
    plt.axis('off')

    plt.subplot(2, 2, 4) 
    plt.imshow(image, cmap='gray')
    plt.title("My image")
    plt.axis('off')


    plt.show()


if __name__ == '__main__':
    part1()
    part2()
    part3()
    part4()
    part5()



