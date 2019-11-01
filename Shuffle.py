
import cv2
import random
labels = 'C:/Users/salehpc/Desktop/ws/Lec 6/Labels/labels.csv'
shuffled_labels = 'C:/Users/salehpc/Desktop/ws/Lec 6/Labels/shuffled_labels.csv'

labelfile = open(labels, "r")
lines = labelfile.readlines()
labelfile.close()
random.shuffle(lines)

shufflefile = open(shuffled_labels, "w")
shufflefile.writelines(lines)
shufflefile.close()