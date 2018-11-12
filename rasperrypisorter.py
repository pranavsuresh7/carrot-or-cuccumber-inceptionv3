#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras 
from keras.models import load_model
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing import image 


# In[2]:


import tensorflow as tf
import RPi.GPIO as GPIO
import time
import picamera


# In[4]:


model = load_model('/home/pi/Downloads/inceptionv3.h5')


# In[ ]:


val = '1'
carrotval = 0
cuccumberval = 0
while(val=='1'):
    
    with picamera.PiCamera() as camera:
        camera.resolution = (800,400)
        camera.start_preview()
        camera.preview_fullscreen=True
        camera.rotation = 180
        # Camera warm-up time
        time.sleep(2)
        camera.capture('sort.jpg')
        img = image.load_img('sort.jpg',target_size=(224,224))
        x = image.img_to_array(img)
        x = preprocess_input(x).reshape(1,224,224,3)
        prediction = model.predict(x)
        
    if(prediction[0,0]>0.5):
        
        cuccumberval= cuccumberval+1
        print('cuccumber',cuccumberval)
    else:
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(12, GPIO.OUT)
        p = GPIO.PWM(12, 50)
        p.start(7.5)
        p.ChangeDutyCycle(7.5)  # turn towards 90 degree
        time.sleep(1) # sleep 1 second
        p.ChangeDutyCycle(2.5)  # turn towards 0 degree
        time.sleep(1) # sleep 1 second
        p.stop()
        GPIO.cleanup()
        carrotval = carrotval+1
        
        print('carrot',carrotval)
    val = input('do you want to continue then press one')


# In[16]:





# In[ ]:




