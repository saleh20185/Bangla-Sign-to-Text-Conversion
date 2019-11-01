import glob

R1 ='E:/FYP/MData/Count/*.jpg'
R2='E:/FYP/MData/Time/*.jpg'
R3='E:/FYP/MData/Tod/*.jpg'
R4='E:/FYP/MData/Req/*.jpg'
R5='E:/FYP/MData/If/*.jpg'

labels = 'C:/Users/salehpc/Desktop/ws/Lec 6/Labels/labels.csv'

objectClass1 = 'Count'
objectClass2 = 'Time'
objectClass3 = 'Today'
objectClass4 = 'Request'
objectClass5 = 'If'

im1 = glob.glob(R1)
im2 = glob.glob(R2)
im3 = glob.glob(R3)
im4 = glob.glob(R4)
im5 = glob.glob(R5)


labelfile = open(labels,'w')

for i in im1:
    labelfile.write(i+','+objectClass1+'\n')
for i in im2:
    labelfile.write(i+','+objectClass2+'\n')
for i in im3:
    labelfile.write(i+','+objectClass3+'\n')
for i in im4:
    labelfile.write(i+','+objectClass4+'\n')
for i in im5:
    labelfile.write(i+','+objectClass5+'\n')

labelfile.close()