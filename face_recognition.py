#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from PIL import Image
import numpy as np
import cv2
import pickle


# In[2]:


image_dir=os.path.join('img')


# In[3]:


face_cascade = cv2.CascadeClassifier('/home/chintan/Projects/Face-recognition--master/haarcascade_frontalface_alt.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()


# In[44]:


current_id=0
label_ids={}
x_train=[]
y_labels=[]


# In[45]:


for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith('jpg') or file.endswith('png'):
            path=os.path.join(root,file)
            label=os.path.basename(os.path.dirname(path))
            if not label in label_ids:
                label_ids[label]=current_id
                current_id+=1
            id_=label_ids[label]
            pil_image=Image.open(path).convert('L')
            image_array=np.array(pil_image,'uint8')
            faces=face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)
            for (x,y,w,h) in faces:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)


# In[46]:


print(y_labels)


# In[47]:


with open('labels.pickle','wb') as f:
    pickle.dump(label_ids,f)


# In[48]:


recognizer.train(x_train,np.array(y_labels))
recognizer.save('trainner.yml')


# In[49]:


label_ids


# In[50]:


len(y_labels)


# In[ ]:





# In[ ]:




