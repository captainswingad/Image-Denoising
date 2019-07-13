#captain swing
#nlm(PATCH WISE)

#######################################################################
#importing basic modules
import numpy as np 
import math
import cv2

########################################################################
#input window size and patch size
windowSize=6
patchSize=5


########################################################################
#getting half width    
delta_window = (int)(windowSize/2)
delta = (int)(patchSize/2)

#input image(PLEASE PROVIDE COMPLETE ADDRESS) 
input_matrix=cv2.imread("C:/lena_test.png")
#print(input_matrix)
#cv2.imshow('input_matrix',input_matrix)

#converting into grayscale
grayscale_input_matrix=cv2.cvtColor(input_matrix,cv2.COLOR_BGR2GRAY)
#saving it into temp variable for further use
inp=grayscale_input_matrix
(fr,fc)=grayscale_input_matrix.shape
#print(gray)
#cv2.imshow('grayscale_input_matrix',grayscale_input_matrix)

#adding noise to it
noise=np.random.normal(0,20,[fr,fc])
grayscale_input_matrix=grayscale_input_matrix+noise
#print(grayscale_input_matrix)
cv2.imshow("grayscale_input_matrix",grayscale_input_matrix)


#padding it
image=np.pad(grayscale_input_matrix,(delta,delta),'reflect').astype(np.float64)
(rows,columns)=image.shape
#print(image)

#defining variable
result=np.zeros((fr,fc))

#######################################################################
#defining kernel..
#used a trick here to calculate gaussian kernel!
#NOTE : just uncomment print statement to understand it! 
a,b = np.mgrid[-delta:delta+1,-delta:delta+1]
#print("1:")
#print(a)
#print("2:")
#print(b)

#input filter parameter and noise variance
h=34
sigma=20

A=((patchSize-1.)/4.)
p=a*a + b*b
#print(p)
w=np.ascontiguousarray(np.exp(-(a*a + b*b)/(2*A*A)).astype(np.float64))
w=1./(np.sum(w)*h*h)*w
#print("3:")
#print(w)

########################################################################
#for understanding given below implementation see report
cnt=0
ii1=0       
for row in range(delta,delta+fr):
    jj1=0
    row_start=row-delta 
    row_end=row+delta+1   
    for col in range(delta,delta+fc):
    	col_start=col-delta
    	col_end=col+delta+1


    	weight_sum=0.0
    	weighted_sum=0.0
    	
    	for i in range(max(-windowSize,delta-row),min(windowSize+1,fr+delta-row)):
    		row_start_i=row_start+i
    		row_end_i=row_end+i
    		for j in range(max(-windowSize,delta-col),min(windowSize+1,fc+delta-col)):
    			col_start_i=col_start+j
    			col_end_i=col_end+j
    			m1=image[row_start:row_end,col_start:col_end]
    			(r1,c1)=m1.shape
    		
    			m2=image[row_start_i:row_end_i,col_start_i:col_end_i]
    			(r2,c2)=m2.shape
    			if r1<3 or r2<3 or c1<3 or c2<3:   #error code if bounadary conditions have some issue
    				print(m1)
    				print(m2)
    				print("---------")
    			threshold=5.0
    			dist=0.0
    			#a=w[delta][delta]*(m1[delta][delta]-m2[delta][delta])*(m1[delta][delta]-m2[delta][delta])
    			#print(a)
    			for ii in range(patchSize):			
    				if dist>threshold:
    					#dist=0
    					break
    				else:
    					for jj in range(patchSize):
    						if ii<r1 and jj<c1 and ii<r2 and jj<c2: #hard code to meet boundary condition 
    							diff=m1[ii][jj]-m2[ii][jj]
    							dist=dist+w[ii][jj]*(diff*diff-2*sigma)
    			dist=max(dist,0)
    			weight=math.exp(-1*dist)
    			weight_sum=weight_sum+weight
    			if row+i<rows and col+j<columns:   
    				weighted_sum=weighted_sum+weight*image[row+i][col+j]



    	result[ii1][jj1]=weighted_sum/weight_sum
    	#print(cnt)
    	cnt=cnt+1
    	jj1=jj1+1
    ii1=ii1+1       

result=result.astype('uint8')                                  
#print(result)

###################################################################

#observation
#calculating psnr ...
psnr=0.0
a=inp-result
a=a*a
print(a)
mse = np.mean(a)
if mse==0:
	psnr=100
else:
	pixel_max=255
	psnr=20*math.log10(pixel_max/math.sqrt(mse))

#print(psnr)


######################################################################
#saving result in text file and writing it
np.savetxt('original.txt',inp)
np.savetxt('nlm.txt',result)

	
#cv2.imwrite("result.png",result)
#cv2.imshow('result',result)

#cv2.waitKey(0)


#####################################end###############################