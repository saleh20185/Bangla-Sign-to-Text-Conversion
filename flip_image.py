import cv2
import numpy as np
#import matplotlib.pyplot as plt


def flip_images():
	gest_folder = "E:/FYP/MData/Req/"
	#images_labels = []
	#images = []
	#labels = []
	for i in range(100):
			path = gest_folder+str(i+1)+".jpg"
			img = cv2.imread(path, 1)
			img1 = cv2.flip(img, 1)#img1 = cv2.GaussianBlur(img,(3,3),0)#img1=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			cv2.imwrite("E:/FYP/MData/Req/"+str(i+1+100)+".jpg", img1)

flip_images()

'''
def flip_images():
    gest_folder = "E:/FYP/MData/cnew"
    for i in range(5):
        path = gest_folder+"/"+str(i+1)+".jpg"
        img = cv2.imread(path, 1)
        M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),90,1)
        dst = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
        cv2.imwrite("E:/FYP/MData/cnewextras/"+str(i+1+5)+".jpg", dst)
        #cv2.imshow('img',dst)
flip_images()
'''
def add_brightness():
    gest_folder = "E:/FYP/MData/tnew"
    for i in range(200):
        path = gest_folder+"/"+str(i+1)+".jpg"
        img = cv2.imread(path, 1)
        #image = cv2.imread('E:/FYP/MData/cnew/195.jpg',cv2.IMREAD_COLOR)
        image_HLS = cv2.cvtColor(img,cv2.COLOR_RGB2HLS) ## Conversion to HLS
        image_HLS = np.array(image_HLS,dtype = np.float64)#,dtype=np.float64) 
        random_brightness_coefficient = np.random.uniform()+0.1 ## generates value between 0.5 and 1.5
        image_HLS[:,:,1] = image_HLS[:,:,1]*random_brightness_coefficient ## scale pixel values up or down for channel 1(Lightness)
        image_HLS[:,:,1][image_HLS[:,:,1]>=255]  = 220 ##Sets all values above 255 to 255
        image_HLS = np.array(image_HLS, dtype = np.uint8)
        image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
        cv2.imwrite("E:/FYP/MData/tnew/"+str(i+1+200)+".jpg", image_RGB)
        #cv2.imshow('dsdas',image_RGB)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
add_brightness()