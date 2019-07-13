#captain_swing
#FNLM(PATCH WISE) 

###############################################################################################################
#importing basic modules
import numpy as np
import cv2
import math
from skimage import measure
import matplotlib.pyplot as plt



################################################################################################################
#input windowSize and patchSize
windowSize=13
patchSize=5

#getting half width
delta_window = (int)(windowSize/2)
delta = (int)(patchSize/2)

#input image(PLEASE PROVIDE COMPLETE ADDRESS)  
input_matrix=cv2.imread("C:/Users/swing captain/Desktop/PYNQ-master/boards/Pynq-Z2/base/standard_test_images/peppers_gray.tif")
#print(input_matrix)
#cv2.imshow('input_matrix',input_matrix)

#converting it into grayscale
grayscale_input_matrix=cv2.cvtColor(input_matrix,cv2.COLOR_BGR2GRAY)
#saving it into temp variable for further use
inp=grayscale_input_matrix
(rr,cc)=grayscale_input_matrix.shape
#print(grayscale_input_matrix)
#cv2.imshow('grayscale_input_matrix',grayscale_input_matrix)

#adding gaussian noise to grayscale input image 
noise=np.random.normal(0,20,[rr,cc])
grayscale_input_matrix=grayscale_input_matrix+noise
#PROBLEM --> SOMETIMES ADDING NOISE WILL CREATE NEGATIVE VALUES(AS GAUSSIAN DISTR. HAS NEGATIVE VALUES) AND IF YOU OPERATE FURTHER ON THAT THAN RESULT WILL CONTAIN WHITE PATCHES 
#SOLUTION --> SAVE IT(COMPRESSION WILL CHANGE NEGATIVE VALUES TO ZERO) AND THEN AGAIN CALL IT!
#saving it into temp for further use
gg=grayscale_input_matrix
cv2.imwrite("ggg.jpg",grayscale_input_matrix) #(PLEASE PROVIDE COMPLETE ADDRESS)
#grayscale_input_matrix=cv2.imread("C:/Users/swing captain/Desktop/PYNQ-master/boards/Pynq-Z2/base/ggg.jpg") #(PLEASE PROVIDE COMPLETE ADDRESS)
#grayscale_input_matrix=cv2.cvtColor(grayscale_input_matrix,cv2.COLOR_BGR2GRAY)
#print(grayscale_input_matrix)
#cv2.imshow(grayscale_input_matrix)


#see report for reasons for padding and for padding size
image=np.pad(grayscale_input_matrix,(delta+delta_window+1,delta+delta_window+1),'reflect').astype(np.float64)
(fr,fc)=image.shape
#image = image.astype('uint8')
#cv2.imshow('image',image)

#defining variables...(see report)
result=np.zeros((fr,fc)).astype(np.float64)
output=np.zeros((rr,cc)).astype(np.float64)
weight_matrix=np.zeros((fr,fc)).astype(np.float64)
integral=np.zeros((fr,fc)).astype(np.float64)


####################################################################################################################
#defining noise variance and filter parameters
#input noise variance sigma
sigma=20
H=[]

#NOTE : IF YOU WANT TO RUN FOR SINGLE filter parameter just give that value in range
#for understanding given below implementation see report
for h in range(25,26):


    H.append(h)
    h2s2=h*h*patchSize*patchSize
    print(h2s2)
    for t_row in range(-delta_window,delta_window+1):
        for t_col in range(0,delta_window+1):
            if t_col==0 and t_row!=0:             #check symmetry
               aplha=0.5                          
            else:
               alpha=1
            for row in range(max(1,-t_row),min(fr,fr-t_row)):
            	for col in range(max(1,-t_col),min(fc,fc-t_col)):
            		t=image[row][col]-image[row+t_row][col+t_col]
            		distance=t*t-sigma*sigma
            		integral[row][col]=distance+integral[row-1][col]+integral[row][col-1]-integral[row-1][col-1]
            for rows in range(max(delta,delta-t_row),min(fr-delta,fr-delta-t_row)):
            	for cols in range(max(delta,delta-t_col),min(fc-delta,fc-delta-t_col)):
            		ssd = integral[rows-delta][cols-delta]+integral[rows+delta][cols+delta]-integral[rows-delta][cols+delta]-integral[rows+delta][cols-delta]
            		#print(ssd)
            		ssd=max(ssd,0)/(h2s2)
            		weight=aplha*2.718**(-1*ssd)
            		weight_matrix[rows][cols]=weight_matrix[rows][cols]+weight
            		weight_matrix[rows+t_row][cols+t_col]=weight_matrix[rows+t_row][cols+t_col]+weight
            		result[rows][cols]=result[rows][cols]+weight*image[rows+t_row][cols+t_col]
            		result[rows+t_row][cols+t_col]=result[rows+t_row][cols+t_col]+weight*image[rows][cols]



    #filling values in output matrix obtained from above
    for r in range(delta,fr-delta):
    	for c in range(delta,fc-delta):
    		if r-delta<rr and c-delta<cc:  #hard code (for safety)
    			output[r-delta][c-delta]=result[r][c]/weight_matrix[r][c]
  
    #print(output)
    #saving image
    cv2.imwrite("output_nlm.jpg",output)
    output = output.astype('uint8')
	#cv2.imshow('output',output)
    

####################################################################################################################
	#observation 

	#finding psnr...
	#output = output.astype(uint8)
    psnr = []
    #finding psnr values for different h values for given noise variance
    a=inp-output
    a=a*a
    mse1 = np.mean(a)
    if mse1==0:
    	psnr.append(100)
    else:
    	pixel_max=255
    	psnr.append(20*math.log10(pixel_max/math.sqrt(mse1)))


    #finding ssim.. 
    s = []
    s.append(measure.compare_ssim(inp,output))

print(psnr)
print(s)

######################################################################################################################
plt.plot(H,psnr,'r--')
plt.xlabel('filter parameter')
plt.ylabel('psnr')
plt.title("dependence of psnr on filter parameter with noise variance as sigma")
plt.legend()
plt.show()

plt.plot(H,s,'b--')
plt.xlabel('filter parameter')
plt.ylabel('ssim')
plt.title("dependence of ssim on filter parameter with noise variance as sigma")
plt.legend()
plt.show()
#####################################################END#############################################################3

