#captain_swing

######################################
#import modules
import numpy as np
import cv2


#######################################

#pre-processing steps

input_matrix=cv2.imread('C:/noisyimg.png')
#print(input_matrix)
'''
cv2.imshow('input_image',input_matrix)
time=cv2.waitKey(0)
if time==27:
    cv2.destroyAllWindows()
'''   
gray_input_matrix=cv2.cvtColor(input_matrix,cv2.COLOR_BGR2GRAY)
#print(gray_input_matrix)
#print(gray_input_matrix.item(2,1))
(image_height,image_width)=gray_input_matrix.shape
output_matrix=np.zeros((image_height,image_width))
#print(image_width)
#print(image_height)
#print(gray_input_matrix.item(479,639))
#####################################################

#user_input
window_height=7 #can be user defined
window_width=7 #can be user defined

#######################################################
#padding
padded_gray_input_matrix=np.pad(gray_input_matrix,((int)((window_height)/2),(int)((window_width)/2)),'symmetric')
#print(padded_gray_input_matrix)
#print(padded_gray_input_matrix.item(4,4))
(new_image_height,new_image_width)=padded_gray_input_matrix.shape
#print(new_image_height)
#print(new_image_width)

########################################################
#implementation
startx=(int)(window_height/2)
starty=(int)(window_width/2)
endx=(int)((new_image_height-starty))
endy=(int)((new_image_width-startx))
#print(padded_gray_input_matrix.item(startx,starty))
#print(padded_gray_input_matrix.item(endx,endy))


temp=[]
xx=0
for x in range(startx,endx):
    yy=0
    for y in range(starty,endy):
        i=0
        for wx in range(x-startx,x+startx):
            for wy in range(y-starty,y+starty):
                temp.append(padded_gray_input_matrix.item(wy,wx))
                #print(padded_gray_input_matrix.item(wx,wy))
                i=i+1
        #print("completed_a_window")
        #print(i)        
        temp.sort()
        #print(temp[(int)((window_height*window_width)/2)])
        output_matrix[yy][xx]=temp[(int)((window_height*window_width)/2)]
        yy=yy+1
        temp=[]            
    xx=xx+1                


#print(output_matrix)
output_matrix=output_matrix.astype("uint8")
print(output_matrix)
print(gray_input_matrix)
#output_matrix=cv2.cvtColor(padded_gray_input_matrix,cv2.COLOR_GRAY2RGB)
#print(padded_gray_input_matrix)
#cv2.imshow('input',gray_input_matrix)
#cv2.waitKey(0)
#cv2.imshow('output',input_matrix)
#cv2.waitKey(0)


##########################################################end######################################################