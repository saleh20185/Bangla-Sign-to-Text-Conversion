import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from classifier_train import CNN_model
from keras import backend as K
K.set_image_data_format('channels_last')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
num_classes = 5
u=0
j=0
i=0
#img = cv2.imread('C:/Users/salehpc/Desktop/ct1.jpg')
#imgRes = cv2.resize(img,(64,64))
cap = cv2.VideoCapture(0) #Webcam Capture
def build_squares(img):
	x, y, w, h = 150, 200, 190,260 
	#d = 5
	imgCrop = None
	crop = None
	cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 2)
			#x+=w+d
	if np.any(crop == None):
			crop = imgCrop
	else:
			crop = np.vstack((crop, imgCrop))
	imgCrop = None
    
		#x = 350
		#y+=h+d
    
	return crop

while(True):
        ret, frame = cap.read()
        org_img =cv2.cvtColor(frame, cv2.IMREAD_COLOR)
        build_squares(org_img)
        imag = cv2.cvtColor(org_img[200:460,150:340],cv2.IMREAD_COLOR)
        img_YCrCb = cv2.cvtColor(imag, cv2.COLOR_BGR2YCrCb)
        #skin color range for hsv color space 
        YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
        YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        if YCrCb_mask.size<img_YCrCb.size:
            imgg=cv2.cvtColor(img_YCrCb,cv2.COLOR_YCrCb2BGR)
            imgRes= cv2.resize(imgg,(128,128))
        
        X_temp = []
        X_temp.append(imgRes)
        X = np.asarray(X_temp)
        
        model = CNN_model()

        #model.load_weights('best.hdf5')

        y = model.predict_classes(X)
        classno = np.ndarray.tolist(y)
        dict = {0: 'Count',1: 'Time',2:'Today',3:'Request',4:'If'}
        objectClass = dict[y[0]]
        print(objectClass)
     
        font = cv2.FONT_HERSHEY_SIMPLEX
        #imag=cv2.resize(img,(300,300))
        cv2.putText(org_img,objectClass,(400,300), font, 1, (200,255,0), 3, cv2.LINE_AA)
        

        cv2.imshow('Sign Prediction',org_img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()