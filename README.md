# Bangla Sign to Text Conversion

In this work, Bangla Sign language to on screen text conversion is done using Convolutional Neural Network(CNN). Due to the unavailability of dataset, 5 Bangla signs are considered and 100 images for each class are taken. Then image data are augmented for feeding this into CNN. A research paper based on this work can be found here: https://ieeexplore.ieee.org/abstract/document/8726895

For the conversion process one should run the files follwing the below sequence:

1. At first 100 images from each five class are extracted from their directory and added labeled into a .csv files according to their respective classes. Run extractor.py for this operation. 

2.Then, images from each class are flipped, added gausian filter and then added brightness using random brightness co-efficient. For this operation run flip_image.py file.

3. Then the labeled are shuffled and thus created a new .csv file named shuffled labels. Run Shuffle.py for this.

4. A separate .npz file is created where lables of all classes are encoded using label encoder. Run npz.py for this.

5. Then images of all classes are fed into CNN in classifier_train.py for training. 

6. Finally,for detection of classes of sign language from real time video with bounding box run classifier.py file. 

