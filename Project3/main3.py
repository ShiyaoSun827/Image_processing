# Import libraries
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import math


def read_image():
    original_img = io.imread('bird.jpeg')
    return original_img


def calculate_trans_mat(image):
    """
    return translation matrix that shifts center of image to the origin and its inverse
    """
    trans_mat = None
    trans_mat_inv = None

    # TODO: implement this function (overwrite the two lines above)
    # ...
    # ...
    h, w = image.shape[:2]
    cy, cx = h/2, w/2
    # trans m to center image to origin
    trans_mat = np.array([[1, 0, -cx],
                          [0, 1, -cy],
                          [0, 0, 1  ]])
    # inverse trans mto shift image back to its original position
    trans_mat_inv = np.array([[1, 0, cx],
                              [0, 1, cy],
                              [0, 0, 1 ]])
    
    return trans_mat, trans_mat_inv



def rotate_image(image):
    ''' rotate and return image '''
    h, w = image.shape[:2]
    trans_mat, trans_mat_inv = calculate_trans_mat(image)

    # TODO: determine angle and create Tr
    angle = ...
    angle_rad = ...
    Tr = np.array([])
    angle = 75
    angle_rad = np.radians(angle)
    Tr = np.array([[np.cos(angle_rad),-np.sin(angle_rad), 0],
                   [np.sin(angle_rad), np.cos(angle_rad), 0],
                   [0,                    0,              1]])  

    # TODO: compute inverse transformation to go from output to input pixel locations
    Tr_inv = np.linalg.inv(Tr)
    Tr_inv_m = trans_mat_inv @ Tr_inv @ trans_mat

    out_img = np.zeros_like(image)
    for out_y in range(h):
        for out_x in range(w):
            # TODO: find input pixel location from output pixel ocation and inverse transform matrix, copy over value from input location to output location
            temp = np.array([out_y, out_x, 1])
            coords = Tr_inv_m @ temp
            
            in_x = int(coords[1])
            in_y = int(coords[0])
            if in_x >= 0 and in_x < w and in_y < h:
                if  in_y >= 0 :
                    out_img[out_y,out_x] = image[in_y,in_x]
    return out_img, Tr_inv
def scale_image(image):
    ''' scale image and return '''
    # TODO: implement this function, similar to above
    out_img = np.zeros_like(image)
    Ts = np.array([])
    h, w = image.shape[:2]

    # Scaling matrix
    Ts = np.array([[2.5, 0, 0],
                   [0, 1.5, 0],
                   [0, 0,   1]])

    Ts_inv = np.linalg.inv(Ts)

    trans_mat, trans_mat_inv = calculate_trans_mat(image)

    Ts_inv_m = trans_mat_inv @ Ts_inv @ trans_mat


# Apply the inverse transformation to each pixel in the output image
    out_img = np.zeros_like(image)
    for out_y in range(h):
        for out_x in range(w):
            # TODO: find input pixel location from output pixel ocation and inverse transform matrix, copy over value from input location to output location
            temp = np.array([out_y, out_x, 1])
            coords = Ts_inv_m @ temp
            
            in_x = int(coords[1])
            in_y = int(coords[0])
            if in_x >= 0 and in_x < w and in_y < h:
                if  in_y >= 0 :
                    out_img[out_y,out_x] = image[in_y,in_x]

    return out_img, Ts_inv



def skew_image(image):
    ''' Skew image and return '''
    # TODO: implement this function like above
    out_img = np.zeros_like(image)
    Tskew = np.array([])
    trans_mat, trans_mat_inv = calculate_trans_mat(image)
    h, w = image.shape[:2]

    #skew parameters
    skew_x = 0.2
    skew_y = 0.2

    # skew trans matrix
    Tskew = np.array([[1, skew_x, 0],
                      [skew_y, 1, 0],
                      [0,      0, 1]])
    
    Tskew_inv = np.linalg.inv(Tskew)
    Tskew_inv_m = trans_mat_inv @ Tskew_inv @ trans_mat
    

    # Initialize output image
    out_img = np.zeros_like(image)
    for out_y in range(h):
        for out_x in range(w):
            # TODO: find input pixel location from output pixel ocation and inverse transform matrix, copy over value from input location to output location
            temp = np.array([out_y, out_x, 1])
            coords = Tskew_inv_m @ temp
            
            in_x = int(coords[1])
            in_y = int(coords[0])
            if in_x >= 0 and in_x < w and in_y < h:
                if  in_y >= 0 :
                    out_img[out_y,out_x] = image[in_y,in_x]


    return out_img, Tskew_inv


def combined_warp(image):
    ''' implement your own code to perform the combined warp of rotate, scale, skew and return image + transformation matrix  '''
    # TODO: implement combined warp on your own. 
    # You need to combine the transformation matrices before performing the warp
    # (you may want to use the above functions to get the transformation matrices)
    out_img = np.zeros_like(image)
    Tc = np.array([])
    h, w = image.shape[:2]
    _, Tr_inv = rotate_image(image)
    _, Ts_inv = scale_image(image)
    _, Tskew_inv = skew_image(image)
    trans_mat, trans_mat_inv = calculate_trans_mat(image)

    Tc = Tskew_inv @ Tr_inv @ Ts_inv
    Tc_m = trans_mat_inv @ Tc @ trans_mat
    out_img = np.zeros_like(image)
    for out_y in range(h):
        for out_x in range(w):
            # TODO: find input pixel location from output pixel ocation and inverse transform matrix, copy over value from input location to output location
            temp = np.array([out_y, out_x, 1])
            coords = Tc_m @ temp
            
            in_x = int(coords[1])
            in_y = int(coords[0])
            if in_x >= 0 and in_x < w and in_y < h:
                if  in_y >= 0 :
                    out_img[out_y,out_x] = image[in_y,in_x]
    
    return out_img, Tc
    

def combined_warp_biinear(image):
    ''' perform the combined warp with bilinear interpolation (just show image) '''
    # TODO: implement combined warp -- you can use skimage.trasnform functions for this part (import if needed)
    # (you may want to use the above functions (above combined) to get the combined transformation matrix)
    out_img = np.zeros_like(image)
    out_img = np.zeros_like(image)
    trans_mat, trans_mat_inv = calculate_trans_mat(image)
    h, w = image.shape[:2]

    angle = 75 


    angle_rad = np.radians(angle)
    Tr_inv = np.array([[np.cos(angle_rad), np.sin(angle_rad), 0],
                      [-np.sin(angle_rad), np.cos(angle_rad), 0],
                      [0,                                 0, 1]])  
    

    Ts_inv = np.array([[1/2.5, 0, 0],
                       [0, 1/1.5, 0], 
                       [0, 0,     1]])

    Tskew_inv = np.array([[1, -0.2, 0],
                          [-0.2, 1, 0],
                          [0,    0, 1]])
    

    Tc = Tskew_inv @ Tr_inv @ Ts_inv
    Tc_inv_m = trans_mat_inv @ Tc @ trans_mat

    for out_y in range(h):
        for out_x in range(w):
            coords = Tc_inv_m @ np.array([out_y, out_x, 1])
            iny = coords[0]
            inx = coords[1]
            ydec, y = math.modf(iny)
            xdec, x = math.modf(inx)
            if (0 <= iny) and (iny < 225) and (0 <= inx) and (inx < 225):
                top = ((1-ydec) * image[int(y)][int(x)] + ydec * image[int(y) + 1][int(x)]) * (1 - xdec)
                bottom = ((1-ydec) * image[int(y)][int(x) + 1] +ydec*image[int(y)][int(x)+1])*xdec
                out_img[out_y][out_x] = bottom + top


    return out_img



if __name__ == "__main__":
    image = read_image()
    plt.imshow(image), plt.title("Oiginal Image"), plt.show()

    rotated_img, _ = rotate_image(image)
    plt.figure(figsize=(15,5))
    plt.subplot(131),plt.imshow(rotated_img), plt.title("Rotated Image")

    scaled_img, _ = scale_image(image)
    plt.subplot(132),plt.imshow(scaled_img), plt.title("Scaled Image")

    skewed_img, _ = skew_image(image)
    plt.subplot(133),plt.imshow(skewed_img), plt.title("Skewed Image"), plt.show()

    combined_warp_img, _ = combined_warp(image)
    plt.figure(figsize=(10,5))
    plt.subplot(121),plt.imshow(combined_warp_img), plt.title("Combined Warp Image")

    combined_warp_biliear_img = combined_warp_biinear(image)
    plt.subplot(122),plt.imshow(combined_warp_biliear_img.astype(np.uint8)), plt.title("Combined Warp Image with Bilinear Interpolation"),plt.show()



