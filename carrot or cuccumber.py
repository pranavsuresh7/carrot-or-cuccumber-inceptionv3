
# coding: utf-8

# # Vegetable Sorting Machine 

# In[2]:


from PIL import Image  # PIL is a module used inputing image from computer
from matplotlib.pyplot import imshow # Showing image to computer
import numpy as np #  computation
import matplotlib.pyplot as plt # Showing image,can be used to plot data


# In[3]:


from os import listdir # listdir returns all the files and directories in the folder 
import tensorflow as tf # for scientific computation
from os.path import isfile, join # used for joining paths of a file or folder 


# # Keras: The Python Deep Learning library     
# Keras is an opensource library for deeplearning. It is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. 

# In[4]:


from keras import layers 
from keras.layers import Input,Dropout, Add, Dense, Activation,GlobalAveragePooling2D, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Sequential
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.initializers import glorot_uniform
from keras.optimizers import Adam
import keras.backend as K


# In[5]:


# getting the image from separate folders for training the neural network
def get_image(file_name): 
    im=Image.open(file_name)
    return im.resize((224,224),Image.ANTIALIAS)
def get_image_folder_list(folder_name):
    return [f for f in listdir(folder_name) if isfile(join(folder_name,f))]
def gettin_array(folder):
    image_list=get_image_folder_list(folder)
    m=[]   
    for i in image_list:
        k=np.array(get_image(folder+i))
     
        m.append(k[np.newaxis:,])
    return m


# In[6]:


kl = get_image_folder_list("/home/pranav/Desktop/carrot_train")
print(kl)
kl = np.array(get_image("/home/pranav/Desktop/carrot_train/images.jpeg"))
print(kl.shape)
train_carrot = np.array(gettin_array("/home/pranav/Desktop/carrot_train/"))
train_y_carrot = np.zeros((train_carrot.shape[0],1)) # creating y as 1 for training datasets
print(train_y_carrot.shape)


# In[7]:


print(train_carrot.shape)
train_cuccumber =  np.array(gettin_array("/home/pranav/Desktop/cuccumber/"))
hjj = get_image_folder_list("/home/pranav/Desktop/cuccumber/") 
train_x = np.concatenate((train_cuccumber,train_carrot),axis=0)
print(train_x.shape)
train_y_cuccumber = np.ones((train_cuccumber.shape[0],1))
print(train_y_cuccumber.shape)


# # pre processing y part of program:
# In order to train the neural network we need to convert Y part ie, cuccumber as one and carrot as zero.
# Then for feeding the full images,concatenate the two classes of images

# In[8]:


train_y = np.concatenate((train_y_cuccumber,train_y_carrot),axis = 0)
print(train_y.shape)


# # Inceptionv3 Convolutional Neural Network Architecture:
# This is deep neural network of several layers. This neural network architecture is trained for ImageNet Large visual recognition using data from 2012. It classify upto 1000 classes of images like vehcles,animals etc.
# ![image.png](attachment:image.png)
# https://arxiv.org/abs/1512.00567
# 
# 

# In[9]:


base_model = InceptionV3(weights='imagenet', include_top=True)


# In[10]:


base_model.summary()


# # Transfer learning:
# Since we don't enough computation power making a deeplearning model with several layers it is always a best choice to use some models which was trained on several different classes of images and retrainig it for our own purpose. 

# In[11]:


for i,layer in enumerate(base_model.layers):
    print(i,layer.name)


# In[12]:


X = base_model.get_layer('conv2d_85').output


# In[13]:


X = GlobalAveragePooling2D()(X)
X = Dense(256,activation='relu')(X)
X = Dense(128,activation='relu')(X)
pred = Dense(1,activation='sigmoid')(X)


# In[14]:


model = Model(inputs=base_model.input,outputs=pred)


# In[15]:


# training only last 44 layers of this deep neural network
for layer in base_model.layers[:268]:
    layer.trainable = False


# In[16]:


model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[17]:


model.fit(train_x,train_y,epochs=10,batch_size =10)


# In[51]:


any_im = train_x[np.random.randint(len(train_x))] # taking a random image from the folder and predicting it
plt.imshow(any_im)
x=image.img_to_array(any_im)
x=preprocess_input(x).reshape(1,224,224,3)
prediction=model.predict(x)


# In[52]:


# the output is a sigmoid function so,the output which is greater than 0.5  threshold is cuccumber and which is less is carrot
if prediction[0,0]>0.5:
    print('cuccumber')
else:
    print('carrot')


# In[164]:


# saving the model to a h5 file inorder run it in rasperrypi
model.save('inceptionv3.h5')

