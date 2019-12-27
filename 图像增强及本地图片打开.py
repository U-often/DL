#!/usr/bin/env python
# coding: utf-8

# In[20]:


import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img


# In[42]:


datagen=ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1/255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img=load_img(r"D:\gequ\111222.jpg")
x=img_to_array(img)
print(x.shape)
x=np.expand_dims(x,0)
print(x.shape)


# In[35]:


i=0
for bach in datagen.flow(x,batch_size=1,save_to_dir='D:\gequ\ggtemp',save_prefix='next_dog',save_format='jpeg'):
    i+=1
    if i==20:
        break

print('finished!')


# In[12]:


#打开图片并显示
plt.figure(figsize=(20,20))
img = plt.imread(r"D:\gequ\111222.jpg")
plt.imshow(img)
plt.axis('off')
plt.show()


# In[ ]:




