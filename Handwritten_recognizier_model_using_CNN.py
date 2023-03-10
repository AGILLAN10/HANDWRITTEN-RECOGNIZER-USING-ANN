#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing libraries
import keras
from keras.datasets import mnist
from keras.models import Sequential 
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as k
import cv2
import os


# In[3]:


#splitting the data
(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[4]:


import matplotlib.pyplot as plt
plt.imshow(x_train[0])


# In[5]:


x_train.shape


# In[6]:


x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)


# In[7]:


input_shape=(28,28,1)


# In[8]:


print(input_shape)


# In[9]:


#converting class vectors into binary
y_train=keras.utils.to_categorical(y_train,10)
y_test=keras.utils.to_categorical(y_test,10)


# In[10]:


x_train=x_train.astype("float32")
x_test=x_test.astype("float32")


# In[11]:


x_train/=255
x_test/=255


# In[12]:


batch_size=128
num_classes=10
epochs=1


# In[35]:


#creating model
model=Sequential()
model.add(Conv2D(10, kernel_size=(2,2),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,kernel_size=(5,5),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))




# In[36]:


#compling the model
model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])


# In[37]:


model.summary()


# In[38]:


#fitting the model
hist=model.fit(x_train,y_train,batch_size=batch_size,epochs=2,verbose=1,validation_data=(x_test,y_test))


# In[39]:


score=model.evaluate(x_test,y_test,verbose=0)


# In[40]:


print("loss",score[0])
print("accuracy",score[1])


# In[41]:


model.save('m_cnn_hr')


# In[43]:


hr_model=load_model('m_cnn_hr')


# In[45]:


image_no=1

while os.path.isfile(f"C:/Users/91936/Documents/Digit_recognizer/testingimg/digits{image_no}.png"):
    print(1)
    try:
        print(2)
        img=cv2.imread(f"C:/Users/91936/Documents/Digit_recognizer/testingimg/digits{image_no}.png")[:,:,0]
        print(3)
        img=np.invert(np.array([img]))
        pred=hr_model.predict(img)
        print(f"The digit is {np.argmax(pred)}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error")
    finally:
        image_no+=1


# In[42]:





# In[32]:





# In[ ]:





# In[ ]:





# In[ ]:



  
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




