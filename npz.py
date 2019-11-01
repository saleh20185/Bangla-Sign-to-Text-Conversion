import cv2
import pandas
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
labels = 'C:/Users/salehpc/Desktop/ws/Lec 6/Labels/labels.csv'
shuffled_labels = 'C:/Users/salehpc/Desktop/ws/Lec 6/Labels/shuffled_labels.csv'
npzfile = 'C:/Users/salehpc/Desktop/ws/Lec 6/Labels/labels.npz'
               
df = pandas.read_csv(shuffled_labels)

rows = df.iterrows()

X_temp = []
Y_temp = []

for row in rows:
    image = row[1][0]
    img = cv2.imread(image)
    img = cv2.resize(img,(96,96)) #(128,128)
    imageClass = row[1][1]
    X_temp.append(img)
    Y_temp.append(imageClass)
    

encoder = LabelEncoder()
encoder.fit(Y_temp)
encoded_Y = encoder.transform(Y_temp)
Y = np_utils.to_categorical(encoded_Y)

np.savez(npzfile, X_train=X_temp,Y_train=Y)