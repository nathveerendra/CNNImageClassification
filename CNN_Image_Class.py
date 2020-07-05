#!/usr/bin/env python
# coding: utf-8

# In[6]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os


# In[24]:


img = image.load_img('train/Jyothsna/1.JPG')
plt.imshow(img)


# In[25]:


img = image.load_img('train/Veeru/2.JPG')
plt.imshow(img)


# In[29]:


cv2.imread("train/Jyothsna/1.JPG").shape


# In[28]:


cv2.imread("train/Veeru/2.JPG").shape


# In[30]:


cv2.imread("train/Jyothsna/1.JPG")


# In[31]:


train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)


# In[97]:


train_ds = train.flow_from_directory("train/",target_size=(80,80),batch_size=1,class_mode='binary')


# In[98]:


validation_ds = validation.flow_from_directory("validation/",target_size=(80,80),batch_size=1,class_mode='binary')


# In[60]:


validation_ds.class_indices


# In[61]:


train_ds.class_indices


# In[62]:


train_ds.classes


# In[63]:


validation_ds.classes


# In[99]:


model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(8,(3,3),activation = 'relu',input_shape = (80,80,3)),
                                      tf.keras.layers.MaxPool2D(2,2),  
                                                             
                                      tf.keras.layers.Conv2D(16,(3,3),activation = 'relu'),
                                      tf.keras.layers.MaxPool2D(2,2),
                                                             
                                      tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                                      tf.keras.layers.MaxPool2D(2,2),
                                                             
                                      tf.keras.layers.Flatten(),
                                      tf.keras.layers.Dense(128,activation = 'relu'),  
                                                             
                                      tf.keras.layers.Dense(1,activation = 'sigmoid')])


# In[100]:


model.compile(loss = 'binary_crossentropy',optimizer=RMSprop(lr=.01),metrics=['accuracy'])


# In[101]:


mdl_fit = model.fit(train_ds,steps_per_epoch =15,epochs=30,validation_data=validation_ds)


# In[104]:


dir_path='testing'
for i in os.listdir(dir_path):
    img = image.load_img(dir_path+"//"+i,target_size=(80,80))
    plt.imshow(img)
    plt.show()
    X = image.img_to_array(img)
    X = np.expand_dims(X,axis=0)
    images=np.vstack([X])
    val=model.predict(images)
    if val == 0:
        print('Jyothsna')
    else:
        print('Veeru')


# In[ ]:




