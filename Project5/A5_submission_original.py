'''
###### DO NOT EDIT ######
(Scroll down for start of the assignment)

# MATLAB Code:
# Alexey S. Sokolov a.k.a. nICKEL, Moscow, Russia
# June 2007
# alex.nickel@gmail.com

Zig-zag section
'''

import numpy as np

# Zigzag scan of a matrix

# --INPUT--
# Argument: 2D matrix of any size, not strictly square 

# --OUTPUT--
# Returns: 1-by-(m*n) array, where input matrix is m*n

def zigzag(input):
    #initializing the variables
    #----------------------------------
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    vmax = input.shape[0]
    hmax = input.shape[1]
    
    #print(vmax ,hmax )

    i = 0

    output = np.zeros(( vmax * hmax))
    #----------------------------------

    while ((v < vmax) and (h < hmax)):
        
        if ((h + v) % 2) == 0:                 # going up
            
            if (v == vmin):
                #print(1)
                output[i] = input[v, h]        # if we got to the first line

                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1                        

                i = i + 1

            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                #print(2)
                output[i] = input[v, h] 
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                #print(3)
                output[i] = input[v, h] 
                v = v - 1
                h = h + 1
                i = i + 1

        
        else:                                    # going down

            if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                #print(4)
                output[i] = input[v, h] 
                h = h + 1
                i = i + 1
        
            elif (h == hmin):                  # if we got to the first column
                #print(5)
                output[i] = input[v, h] 

                if (v == vmax -1):
                    h = h + 1
                else:
                    v = v + 1

                i = i + 1

            elif ((v < vmax -1) and (h > hmin)):     # all other cases
                #print(6)
                output[i] = input[v, h] 
                v = v + 1
                h = h - 1
                i = i + 1


        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
            #print(7)        	
            output[i] = input[v, h] 
            break

    #print ('v:',v,', h:',h,', i:',i)
    return output


# Inverse zigzag scan of a matrix

# --INPUT--
# Argument: 1-by-m*n array, m & n are vertical & horizontal sizes of output matrix

# --OUTPUT--
# Returns: a 2D matrix of defined sizes with input array items gathered by zigzag

def inverse_zigzag(input, vmax, hmax):

    #print input.shape

    # initializing the variables
    #----------------------------------
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    output = np.zeros((vmax, hmax))

    i = 0
    #----------------------------------

    while ((v < vmax) and (h < hmax)): 
        #print ('v:',v,', h:',h,', i:',i)   	
        if ((h + v) % 2) == 0:                 # going up
            
            if (v == vmin):
                #print(1)
                
                output[v, h] = input[i]        # if we got to the first line

                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1                        

                i = i + 1

            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                #print(2)
                output[v, h] = input[i] 
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                #print(3)
                output[v, h] = input[i] 
                v = v - 1
                h = h + 1
                i = i + 1

        
        else:                                    # going down

            if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                #print(4)
                output[v, h] = input[i] 
                h = h + 1
                i = i + 1
        
            elif (h == hmin):                  # if we got to the first column
                #print(5)
                output[v, h] = input[i] 
                if (v == vmax -1):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
                                
            elif((v < vmax -1) and (h > hmin)):     # all other cases
                output[v, h] = input[i] 
                v = v + 1
                h = h - 1
                i = i + 1


        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
            #print(7)        	
            output[v, h] = input[i] 
            break


    return output


'''
######
Assignment 5 starts here
######
'''

def part1_encoder():
	# JPEG encoding

    import numpy as np
    # import scipy
    import matplotlib.pyplot as plt
    from skimage import io
    from scipy.fftpack import dct,idct  


    # NOTE: Defining block size
    block_size = 8

    # TODO: Read image using skimage.io
    ###### Your code here ######

    plt.imshow(img)
    plt.title('input image (RGB)')
    plt.axis('off')
    plt.show()
    
    # TODO: Convert the image from RGB space to YCbCr space
    
    plt.imshow(np.uint8(img))
    plt.title('input image (YCbCr)')
    plt.axis('off')
    plt.show()


    '''
    Interesting property: Separability

    The separability property refers to the fact that a 2D DCT can be computed as the product of two 1D DCTs 
    applied along each dimension of the data independently. This means that a 2D DCT can be computed much more 
    efficiently as two 1D DCTs instead of directly computing the 2D transform.
    '''

    # TODO: Function to compute 2D Discrete Cosine Transform (DCT)
    # Apply DCT with type 2 and 'ortho' norm parameters

    def dct2D(x):
        ###### Your code here ######
        return result


    # TODO: Get size of image
    ###### Your code here ######

    # TODO: Compute number of blocks (of size 8-by-8), cast the numbers to int

    nbh = ###### Your code here ###### # (number of blocks in height)
    nbw = ###### Your code here ###### # (number of blocks in width)


    # TODO: (If necessary) Pad the image, get size of padded image
    H = ###### Your code here ######  # height of padded image
    W = ###### Your code here ######  # width of padded image

    # TODO: Create a numpy zero matrix with size of H,W,3 called padded img
    padded_img = ###### Your code here ######

    # TODO: Copy the values of img into padded_img[0:h,0:w,:]
    ###### Your code here ######

    # TODO: Display padded image
    plt.imshow(np.uint8(padded_img))
    plt.title(###### Title here ######)
    plt.axis('off')
    plt.show()


    # TODO: Create the quantization matrix
    # Refer to this https://www.sciencedirect.com/science/article/pii/S1742287608000285#:~:text=In%20the%20quantization%20stage%2C%20the,range%20from%200%20to%2065%2C535.&text=A%20lower%20number%20means%20that,higher%20quality%20image%20should%20result.
    # Use Fig. 1 as your quantization tables (1 table for Y channel and 1 table for Cb and Cr channels)
    quantization_matrix_Y = # quantization table for Y channels
    quantization_matrix_CbCr = # quantization table for Cb and Cr channels
    ###### Your code here ######

    # TODO: Initialize variables for compression calculations (only for the Y channel)
    ###### Your code here ######

    # NOTE: Iterate over blocks
    for i in range(nbh):
        
        # Compute start and end row indices of the block
        row_ind_1 = i * block_size
        row_ind_2 = row_ind_1 + block_size
        
        for j in range(nbw):
            
            # Compute start and end column indices of the block
            col_ind_1 = j * block_size 
            col_ind_2 = col_ind_1 + block_size
            
            # TODO: Select current block to process using calculated indices (through slicing)
            Yblock = ###### Your code here ######
            Cbblock = ###### Your code here ######
            Crblock = ###### Your code here ######
            
            # TODO: Apply dct2d() to selected block             
            YDCT = ###### Your code here ######
            CbDCT = ###### Your code here ######
            CrDCT = ###### Your code here ######

            # TODO: Quantization
            # Divide each element of DCT block by corresponding element in quantization matrix
            quantized_YDCT = ###### Your code here ######
            quantized_CbDCT = ###### Your code here ######
            quantized_CrDCT = ###### Your code here ######

            # TODO: Reorder DCT coefficients into block (use zigzag function)
            reordered_Y = ###### Your code here ######
            reordered_Cb = ###### Your code here ######
            reordered_Cr = ###### Your code here ######

            # TODO: Reshape reordered array to 8-by-8 2D block
            reshaped_Y = ###### Your code here ######
            reshaped_Cb = ###### Your code here ######
            reshaped_Cr = ###### Your code here ######

            # TODO: Copy reshaped matrix into padded_img on current block corresponding indices
            ###### Your code here ######

            # TODO: Compute pixel locations with non-zero values before and after quantization (only in Y channel)
            # TODO: Compute total number of pixels
            ###### Your code here ####

    plt.imshow(np.uint8(padded_img))
    plt.title('encoded image')
    plt.axis('off')
    plt.show()
                

    # NOTE: Write h, w, c, block_size and padded_img into .txt files at the end of encoding

    # TODO: Write padded_img into 'encoded.npy' file (use np.save() function)
    # First parameter should be 'encoded.npy'
    ###### Your code here ######

    # TODO: write [h, w, c, block_size] into size.txt
    # First parameter should be 'size.txt' (use np.savetxt() function)
    ###### Your code here ######

    # TODO: Calculate percentage of pixel locations with non-zero values before and after to measure degree of compression 

    # Print statements as shown in eClass
    ###### Your code here ######




def part2_decoder():
    # JPEG decoding

    import numpy as np
    # import scipy
    import matplotlib.pyplot as plt
    from skimage import io
    from scipy.fftpack import dct,idct 

    # NOTE: Defining block size
    block_size = 8 

    # TODO: Function to compute 2D Discrete Cosine Transform (DCT)
    # Apply IDCT with type 2 and 'ortho' norm parameters

    def idct2D(x):
        ###### Your code here ######
        return result


    # TODO: Load 'encoded.npy' into padded_img (using np.load() function)
    ###### Your code here ######

    # TODO: Load h, w, c, block_size and padded_img from the size.txt file
    ###### Your code here ######

    # TODO: 6. Get size of padded_img, cast to int if needed
    ###### Your code here ######

    # TODO: Create the quantization matrix (Same as before)
    quantization_matrix_Y = # quantization table for the Y channel
    quantization_matrix_CbCr = # quantization table for the Y channel
    
    ###### Your code here ######

    # TODO: Compute number of blocks (of size 8-by-8), cast to int
    nbh = ###### Your code here ###### # (number of blocks in height)
    nbw = ###### Your code here ###### # (number of blocks in width)

    # TODO: iterate over blocks
    for i in range(nbh):
        
            # Compute start and end row indices of the block
            row_ind_1 = i * block_size
            
            row_ind_2 = row_ind_1 + block_size
            
            for j in range(nbw):
                
                # Compute start and end column indices of the block
                col_ind_1 = j * block_size

                col_ind_2 = col_ind_1 + block_size
                
                # TODO: Select current block to process using calculated indices
                Yblock = ###### Your code here ######
                Cbblock = ###### Your code here ######
                Crblock = ###### Your code here ######
                
                # TODO: Reshape 8-by-8 2D block to 1D array
                Yreshaped = ###### Your code here ######
                Cbreshaped = ###### Your code here ######
                Crreshaped = ###### Your code here ######
                
                # TODO: Reorder array into block (use inverse_zigzag function)
                Yreordered = ###### Your code here ######
                Cbreordered = ###### Your code here ######
                Crreordered = ###### Your code here ######
                
                # TODO: De-quantization
                # Multiply each element of reordered block by corresponding element in quantization matrix
                dequantized_YDCT = ###### Your code here ######
                dequantized_CbDCT = ###### Your code here ######
                dequantized_CrDCT = ###### Your code here ######
                
                # TODO: Apply idct2d() to reordered matrix 
                YIDCT = ###### Your code here ######
                CbIDCT = ###### Your code here ######
                CrIDCT = ###### Your code here ######

                # TODO: Copy IDCT matrix into padded_img on current block corresponding indices
                ###### Your code here ######

    # TODO: Remove out-of-range values
    ###### Your code here ######

    plt.imshow(np.uint8(padded_img))
    plt.title('decoded padded image (YCbCr)')
    plt.axis('off')
    plt.show()

    # TODO: Get original sized image from padded_img
    ###### Your code here ######

    plt.imshow(np.uint8(decoded_img))
    plt.title('decoded padded image (YCbCr)')
    plt.axis('off')
    plt.show()
    
    # TODO: Convert the image from YCbCr to RGB
    ###### Your code here ######
    
    # TODO: Remove out-of-range values
    ###### Your code here ######
    
    plt.imshow(np.uint8(decoded_img))
    plt.title('decoded image (RGB)')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    part1_encoder()
    part2_decoder()
