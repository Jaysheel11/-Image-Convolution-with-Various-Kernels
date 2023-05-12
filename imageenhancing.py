#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import scipy as sp
                                   
import itertools as i
import matplotlib. pyplot as plt
from scipy import signal
from matplotlib. animation import FuncAnimation
import os


# In[7]:


# import an image
FNAME = "yourimage.jpg"
         
# inspect image data
im_data  = plt.imread(FNAME)
print (im_data)
print(im_data.shape)
print(im_data.size)
# rescale image values - if needed
im_data = plt.imread(FNAME).astype(float)/255.
print(im_data)


# In[8]:


# define/create convoluation function

def im_convolve(im1, kern): # function takes 2 arguments: image and kernel
    im2 = np.empty_like(im1) # create an empty image to store filtered image
    for dim in range(im1.shape[-1]): # Loop over rgb channels
        im2[:, :,dim] = sp.signal.convolve2d(im_data[:, :, dim],
                                            kern,
                                            mode="same",
                                            boundary="symm")
    return im2 # return filtered image


# In[9]:


# create kernel matrices

KERNELS = {"Edge Detection 3x3": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
           "Sharpen 3x3": np.array([[0.01, -1, 0.01], [-1, 5, -1], [0.01, -1, 0.01]]),
           "G_Blur 3x3": np.array([[.06, .12, .06], [.12, .24, .12], [.06, .12, .06]]),
           "Blur 3x3": np.array([[.11,.11,.11], [.11,.11,.12], [.11,.11,.11]]),
           "Edge2 3x3": np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])}

   

   

   
kernel_name ="Edge Detection 3x3"
  

kernel_name = "Blur 3x3"
kernel_name = "Edge2 3x3"
kernel_name = "G_Blur 3x3"
kernel_name = "Sharpen 3x3"

kernel = KERNELS[kernel_name]



# In[10]:


# run convolution and display images
im_filtered = im_convolve(im_data, kernel)

plt.rcParams["figure.figsize"] = (12, 7)

fig, (axL, axR) = plt.subplots(ncols=2, tight_layout=True)
fig.suptitle(kernel_name)

 

axL.imshow(im_data)
axR.imshow(im_filtered)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




