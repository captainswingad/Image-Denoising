#captain_swing

#importing basic modules
##################################################################
import numpy as np
import cv2

###################################################################
#pre-processing steps

input_matrix=cv2.imread("C:/lenna.jpg")
#print(input_matrix)
grayscale_input_matrix=cv2.cvtColor(input_matrix,cv2.COLOR_BGR2GRAY)
#print(grayscale_input_matrix)
(image_height,image_width)=grayscale_input_matrix.shape
#print(image_height)
#print(image_width)
output_matrix=np.zeros((image_height,image_width),int)
#print(output_matrix)

#####################################################################

#input sigma
sigma=0.84089642
#print(sigma)

#####################################################################
#defining kernel

import math
kernel_height=7
kernel_width=7
kernel_matrix=np.zeros([kernel_height,kernel_width],float)
print(kernel_matrix)
norm_factor=0


#implementing kernel
for i in range(0,kernel_width):
    for j in range(0,kernel_height):
        a=(int)(kernel_width/2)
        #print(a)
        b=(int)(kernel_height/2)
        #print(b)
        sqr_dist=(i-a)*(i-a) + (j-b)*(j-b)
        #print(sqr_dist)          
        kernel_matrix[j,i]=math.exp((-1*sqr_dist)/(2*sigma*sigma))
        norm_factor+=kernel_matrix[j,i]     


kernel_matrix=kernel_matrix/norm_factor
#print(kernel_matrix)

############################################################################

#padding step...

padded_gray_matrix=np.pad(grayscale_input_matrix,((int)(kernel_height/2),(int)(kernel_width/2)),'symmetric')
#print(padded_gray_matrix)
#print(padded_gray_matrix.shape)

##############################################################################

#implementing gaussian filter

startx=(int)(kernel_width/2)
starty=(int)(kernel_height/2)
(new_image_height,new_image_width)=padded_gray_matrix.shape
endx=new_image_width-startx-1
endy=new_image_height-starty-1
#print(startx)
#print(starty)
#print(endx)
#print(endy)
#print(padded_gray_matrix[starty,startx])
#print(padded_gray_matrix[endy,endx])
i=0
for x in range(startx,endx+1):
    j=0
    for y in range(starty,endy+1):
        sume=0
        ki=0
        for wx in range(x-startx,x+startx+1):
            kj=0
            for wy in range(y-starty,y+starty+1):
                sume+=kernel_matrix[kj,ki]*padded_gray_matrix[wy,wx]
                kj=kj+1
            ki=ki+1
        output_matrix[j][i]=(int)(sume)
        j=j+1
    i=i+1

#print(output_matrix.shape)
######################################################################

#library gaussian blur
blur = cv2.GaussianBlur(grayscale_input_matrix,(7,7),0.84089642)
#print(blur)

#cv2.imshow('img',blur)
#cv2.waitKey(0)

#####################################################end###############################################



