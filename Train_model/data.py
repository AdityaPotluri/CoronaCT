import os
import numpy as np
import cv2
import random
import pickle

categories=['CT_COVID','CT_NonCOVID']


dir=r"C:\Users\adity\Desktop\Programming\Python\CoronaCT\Dataset"

IMG_HEIGHT=100
IMG_WIDTH=140

training_data=[]
for category in categories:
    path=os.path.join(dir,category)
    category_num=categories.index(category)
    for img in os.listdir(path):
        try:
            img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            img_array=cv2.resize(img_array,(IMG_HEIGHT,IMG_WIDTH))
           
            training_data.append([img_array,category_num])
        except Exception as e:
            pass

random.shuffle(training_data)
        
X=[]
y=[]

for features,labels in training_data:
    X.append(features)
    y.append(labels)

X=np.array(X).reshape(-1,IMG_HEIGHT,IMG_WIDTH,1)
print(X.shape)
pickle_out=open("X.pickle","wb")
pickle.dump(X,pickle_out)

pickle_out=open("y.pickle","wb")
pickle.dump(y,pickle_out)

pickle_out.close()