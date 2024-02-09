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
    plt.title("Laplace filterd image")
    plt.show()

    plt.subplot(1, 2, 1) 
    plt.imshow(source_gs, cmap='gray')
    plt.title("Original")
    Gaussian = filtering(source_gs,'Gaussian')
    plt.subplot(1, 2, 2) 
    plt.imshow(Gaussian, cmap='gray')
    plt.title(" filterd image ")
    plt.show()

    plt.subplot(1, 2, 1) 
    plt.imshow(source_gs, cmap='gray')
    plt.title("Original")
    filterd = filtering(source_gs,'filterd')

    plt.subplot(1, 2, 2) 
    plt.imshow(filterd, cmap='gray')
    plt.title(" filterd image ")
    plt.show()

    plt.subplot(1,2,1)
    plt.imshow((source_gs).astype(int), cmap='gray', vmin=0, vmax=255)
    plt.title("Cameraman")
    plt.subplot(1,2,2)
    out = filtering(source_gs,'Enhance')
    plt.imshow((source_gs+out).astype(int), cmap='gray', vmin=0, vmax=255)
    plt.title("Enhanced Cameraman")






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
        #padding first,get the h value and w value
        h_kernel = kernel.shape[0]
        w_kernel = kernel.shape[1]
        h_padding = np.cast['int']((h_kernel - 1.) / 2.)
        w_padding = np.cast['int']((w_kernel - 1.) / 2.)
        #pad_img = np.zeros((source_gs.shape[0] + h_kernel*2,source_gs.shape[1] + w_kernel*2))
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
        '''
        for i in np.arange(H,H-h_kernel,1):
            for j in np.arange(W,W-w_kernel,1):
                for m in np.arange(-h_kernel,h_kernel+1,1):
                    for n in np.arange(-w_kernel,w_kernel+1,1):
                        output[i,j] += source_gs[i+m,j+n] * kernel[m+h_kernel,n+w_kernel]
        '''
        for i in np.arange(h_padding,h-h_padding,1):
            for j in np.arange(w_padding,w-w_padding,1):
                for m in np.arange(-h_padding,h_padding+1,1):
                    for n in np.arange(-w_padding,w_padding+1,1):
                        output[i,j] += source_gs[i+m,j+n] * kernel[m+h_padding,n+w_padding]
        return output
    elif filter == 'Gaussian':
        
       
        h = source_gs.shape[0]
        w = source_gs.shape[1]
        h_image,w_image = source_gs.shape
        gker = np.array([[0,0,0,0,0], [0,1,0,1,0],[0,0,0,1,0]]) 

        h_ker,w_ker = gker.shape 
    
        hf_ker = np.cast['int']((h_ker-1.)/2.) 
        wf_ker = np.cast['int']((w_ker-1.)/2.) 
        
        padding = np.zeros((h + hf_ker * 2 , w + w_ker *2 )) 
        output_padding = np.zeros((h + hf_ker * 2 ,w + w_ker *2))

        padding[hf_ker:(h+hf_ker), wf_ker:(w+wf_ker)] = source_gs #

    
        for i in np.arange(hf_ker,h_image-hf_ker,1):
            for j in np.arange(wf_ker,w_image-wf_ker,1):
                for l in np.arange(-hf_ker,hf_ker+1,1):
                    for m in np.arange(-wf_ker,wf_ker+1,1):
                        output_padding[i,j] += source_gs[i+l,j+m]*gker[l+hf_ker,m+wf_ker]
        return output_padding
    elif filter == 'filterd':
        h = source_gs.shape[0]
        w = source_gs.shape[1]
        h_image,w_image = source_gs.shape
        
        gker = np.array([[0,0,0], [6,0,6],[0,0,0]]) 

        h_ker,w_ker = gker.shape 
    
        hf_ker = np.cast['int']((h_ker-1.)/2.) 
        wf_ker = np.cast['int']((w_ker-1.)/2.)
        
        padding = np.zeros((h + hf_ker * 2 , w + w_ker *2 )) 
        output_padding = np.zeros((h + hf_ker * 2 ,w + w_ker *2))

        padding[hf_ker:(h+hf_ker), wf_ker:(w+wf_ker)] = source_gs 

    
        for i in np.arange(hf_ker,h_image-hf_ker,1):
            for j in np.arange(wf_ker,w_image-wf_ker,1):
                for l in np.arange(-hf_ker,hf_ker+1,1):
                    for m in np.arange(-wf_ker,wf_ker+1,1):
                        output_padding[i,j] += source_gs[i+l,j+m]*gker[l+hf_ker,m+wf_ker]
        return output_padding
    elif filter == 'Enhance':
        h = source_gs.shape[0] 
        w = source_gs.shape[1]
        h_image,w_image = source_gs.shape
    
        h_image,w_image = source_gs.shape 
        gker = np.array([[0, -1, 0], [-1,4,-1],[0,-1,0]]) 
        h_ker,w_ker = gker.shape 
    
        hf_ker = np.cast['int']((h_ker-1.)/2.) 
        wf_ker = np.cast['int']((w_ker-1.)/2.) 
        
        padding = np.zeros((h + hf_ker * 2 , w + w_ker *2 )) 
        source_gs = source_gs.astype('float64')
        output_padding = np.zeros_like(source_gs)

        padding[hf_ker:(h+hf_ker), wf_ker:(w+wf_ker)] = source_gs

    
        for i in np.arange(hf_ker,h_image-hf_ker,1):
            for j in np.arange(wf_ker,w_image-wf_ker,1):
                for l in np.arange(-hf_ker,hf_ker+1,1):
                    for m in np.arange(-wf_ker,wf_ker+1,1):
                        output_padding[i,j] += source_gs[i+l,j+m]*gker[l+hf_ker,m+wf_ker]
        return output_padding







def part2():
    """add your code here"""
    noisy = io.imread("noisy.jpg") 

    height = noisy.shape[0] 
    width = noisy.shape[1]


    temp_noisy = noisy
    #apply median filter to remove the noise
    image_processed_by_median = cv2.medianBlur(temp_noisy, 5)#apply the medianBlur filter
    #save the image  
    mean_image = "mean_image.jpg"
    cv2.imwrite(mean_image,image_processed_by_median)
 
   
    #Apply a Gaussian filter to the same noisy image.
    '''
    We should specify the width and height of the kernel which should be positive and odd.
    We also should specify the standard deviation in the X and Y directions, 
    sigmaX and sigmaY respectively. 
    If only sigmaX is specified, sigmaY is taken as the same as sigmaX. 
    If both are given as zeros, they are calculated from the kernel size. 
    Gaussian blurring is highly effective in removing Gaussian noise from an image.
    '''
    image_processed_by_Guassian = cv2.GaussianBlur(temp_noisy,(5,5),0) 
    #save the image
    Gaussian_image = "Gaussian_image.jpg"
    cv2.imwrite(Gaussian_image,image_processed_by_Guassian)
    
    #output three images
    plt.subplot(1, 3, 1) 
    plt.imshow(noisy, cmap='gray')
    plt.title("Original")
    
    plt.subplot(1, 3, 2) 
    plt.imshow(image_processed_by_median, cmap='gray')
    plt.title("median")

    plt.subplot(1, 3, 3) 
    plt.imshow(image_processed_by_Guassian, cmap='gray')
    plt.title("Gaussian")
    plt.show()


def part3():
    """add your code here"""
    damage_camera = io.imread("damage_cameraman.png") 
    damage_mask = io.imread("damage_mask.png")
    temp_damage = io.imread("damage_cameraman.png")

    height = damage_camera.shape[0] 
    width =  damage_camera.shape[1]

    damage_pixel = [] #Create an empty list to store the pixel coordinates that need to be repaired.

    for i in range(height): # Iterate over the rows of the image.
      for j in range(width):# Iterate over the columns of the image.
        if damage_mask[i][j].all() == 0: #Check if the pixel at (i,j) in the mask image is damaged. If it is damaged (all channels are 0), then add its coordinates to the damage_pixel list.
          damage_pixel.append((i,j)) # If the pixel is damaged, append its coordinates to the damage_pixel list.
    
    i = 1  #Initialize the variable i to 1.
    while(i<= 5000): #  Loop 5000 times.
      Gaussian_image = cv2.GaussianBlur(temp_damage,(5,5),0)#J = GaussianSmooth(J)  Smooth damaged image
      #Apply a Gaussian filter to the damaged image using cv2.GaussianBlur() function. The (5,5) argument specifies the size of the kernel, and 0 specifies the standard deviation of the filter along both axes.
      for j in damage_pixel: 
        temp_damage[j] = Gaussian_image[j] 
       
      i = i + 1 
            
    plt.subplot(1, 2, 1) 
    plt.imshow(damage_camera, cmap='gray')
    plt.title("damaged image")

    plt.subplot(1, 2, 2) 
    plt.imshow(temp_damage, cmap='gray')
    plt.title("restored image")

    plt.show()



def part4():
    """add your code here"""
    ex2 = io.imread("ex2.jpg")

    height = ex2.shape[0] 
    width =  ex2.shape[1] 

    h_image,w_image = ex2.shape 
    gker = np.array([[-1, 0, 1], [-2,0,2],[-1,0,1]]) #input kernal
    h_ker,w_ker = gker.shape 
   

    hf_ker = np.cast['int']((h_ker-1.)/2.) 
    wf_ker = np.cast['int']((w_ker-1.)/2.) # 
    

    padding = np.zeros((height + hf_ker * 2 , width + w_ker *2 )) 
    output_padding = np.zeros((height + hf_ker * 2 ,width + w_ker *2))

    padding[hf_ker:(height+hf_ker), wf_ker:(width+wf_ker)] = ex2 
    for i in np.arange(hf_ker,h_image-hf_ker,1):    
        for j in np.arange(wf_ker,w_image-wf_ker,1):
            for l in np.arange(-hf_ker,hf_ker+1,1):
                for m in np.arange(-wf_ker,wf_ker+1,1):
                    output_padding[i,j] += ex2[i+l,j+m]*gker[l+hf_ker,m+wf_ker] 

    h_image,w_image = ex2.shape #unpacks the dimensions of the input image ex2 and assigns them to the variables h_image and w_image respectively, 
                                #which represent the height and width of the image in pixels.
    gker_vertical = np.array([[1, 2, 1], [0,0,0],[-1,-2,-1]]) #input kernal
    h_ker,w_ker = gker.shape 
   

    hf_ker = np.cast['int']((h_ker-1.)/2.) 
    wf_ker = np.cast['int']((w_ker-1.)/2.) # 
    
    padding = np.zeros((height + hf_ker * 2 , width + w_ker *2 )) 
    output_padding_vertical = np.zeros((height + hf_ker * 2 ,width + w_ker *2))

    padding[hf_ker:(height+hf_ker), wf_ker:(width+wf_ker)] = ex2 

    for i in np.arange(hf_ker,h_image-hf_ker,1):   
        for j in np.arange(wf_ker,w_image-wf_ker,1):
            for l in np.arange(-hf_ker,hf_ker+1,1):
                for m in np.arange(-wf_ker,wf_ker+1,1):
                    output_padding_vertical[i,j] += ex2[i+l,j+m]*gker[l+hf_ker,m+wf_ker] 


    E = np.sqrt(output_padding**2 + output_padding_vertical**2) # Edge strength or Gradient magnitude
 
    plt.figure(figsize=(15,7))
    plt.subplot(4, 1, 1) 
    plt.imshow(ex2, cmap='gray')
    plt.title("image")

    plt.subplot(4, 1, 2) 
    plt.imshow(output_padding, cmap='gray')
    plt.title("Horizontal")

    plt.subplot(4, 1, 3 ) 
    plt.imshow(output_padding_vertical, cmap='gray')
    plt.title("Vertical")

    plt.subplot(4, 1, 4 ) 
    plt.imshow(E, cmap='gray')
    plt.title("Gradient")

    plt.show()


def part5():
    """add your code here"""
    picture = io.imread("ex2.jpg", as_gray=True) 
    target = io.imread("canny_target.jpg", as_gray=True) 
    best_distance = 100000 
    best_params = [0,0,0]  
    for low_thresh in range(50,100, 20):
        for high_threshold in range(100,200,15):
            for sigma in np.arange(1.0,3.0,0.2):
                canny = feature.canny(image=picture, sigma=sigma, low_threshold=low_thresh, high_threshold=high_threshold) 
                # Apply the Canny method and the parameters to the image
                this_dist = distance.cosine(canny.flatten(),target.flatten())
                if this_dist < best_distance:
                    best_distance = this_dist
                    best_params = [sigma,low_thresh,high_threshold]
    my_image = feature.canny(image = picture, sigma=best_params[0],low_threshold=best_params[1],high_threshold=best_params[2])
    #feature.canny is a function that implements the Canny edge detection algorithm
    print(best_distance)
    print(best_params)


    image_processed_by_Guassian = cv2.GaussianBlur(picture,(5,5),0) # This code applies a Gaussian filter to the original image picture using the OpenCV GaussianBlur() function, which takes the image and the size of the kernel as input. 
                                #(5,5) in this case is the size of the Gaussian kernel used for blurring the image. 
                                #The 0 parameter is the standard deviation of the Gaussian kernel, indicating that it is auto-calculated based on the kernel size.
    #save the image
    Gaussian_image = "Gaussian_image.jpg"
    cv2.imwrite(Gaussian_image,image_processed_by_Guassian)
      
                
    plt.subplot(4, 1, 1) 
    plt.imshow(picture, cmap='gray')
    plt.title("image")

    plt.subplot(4, 1, 2) 
    plt.imshow(image_processed_by_Guassian , cmap='gray')
    plt.title("Guassian")


    plt.subplot(4, 1, 3) 
    plt.imshow(target, cmap='gray')
    plt.title("target")

    plt.subplot(4, 1, 4) 
    plt.imshow(my_image, cmap='gray')
    plt.title("my_image")


    plt.show()


if __name__ == '__main__':
    part1()
    part2()
    part3()
    part4()
    part5()



