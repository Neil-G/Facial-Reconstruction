# -*- coding: utf-8 -*-
"""
@author: neil

Script Info:
The image data consists of 3 stacks of 64x32 facial images. The training set 
consists of 12,000 left and right images each that go together to form a single 
64x64 image. The test set contains 1,233 images of only the left side of the 
image. The goal is to reconstruct the right side of the test images using a 
filter trained by the training data

1. Creates a Facial_Reconstruction folder and downloads the image data into this folder
2. Converts the 3-d stacks of images into 2-d matrices of image vectors
3. Creates a filter using least squares regression
4. Runs the test images through the filter to reconstruct the missing parts of the images
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import random 
import scipy.io
import urllib


os.mkdir(os.getcwd() + '/Facial_Reconstruction')    # make a directory called Facial_Reconstruction
os.chdir('Facial_Reconstruction')                   # set working directory to the direct
print 'Image data files will be downloaded to {}'.format(os.getcwd())   # show location data is downloading to
data_location = "http://www.sci.ccny.cuny.edu/~szlam/2013-fall-366/"    # url location of data
image_stack_names = ['train_data_left',                                 # names of the data matrices
                     'train_data_right',
                     'test_data_left']
img_stacks = []  # this list will hold the 3 matrices containing image data
for index, name in enumerate(image_stack_names):
    file_name = name + '.mat'
    urllib.urlretrieve(data_location + file_name, file_name)       # download the file to the Facial Recognition folder
    print 'File {} of 3 has finished downloading'.format(index+1)  # send a message that the matrix data downloaded successfully
    img_stacks.append(scipy.io.loadmat(name)[name])         # convert the matlab data to a numpy array 
                                                            # and store it in data as a 3-dimensional array
                                           
# We now have the image data in 3-dimensional stacks of 64x32 images
# We need to create a 2-dimensional arrays out of these stacks where each row represents a single image                                           
def convert_to_rows(stack):
    '''Goes through a stack of images and creates a numpy matrix where every row is an image'''
    size = stack.shape[2]           # get number of pictures in stacks
    images = np.empty((size, 2048)) # create empty matrix to hold image vectors
    for i in range(size):           # create loop to populate matrix with image vectors
        images[i] = stack[:,:,i].reshape(2048,)     # reshape image to a vector and store it in the matrix
    return images

img_rows = []                            # this list will hold the image vectors
for stack in img_stacks:
    img_rows.append(convert_to_rows(stack))  # convert the stacked images into vector images 
    
left, right, test = img_rows                # unpack our images in convenient names   
weights = np.linalg.lstsq(left, right)[0]   # use least squares regression to find weight matrix that reconstructs partial facial images
reconstructed = np.dot(test, weights)       # then reconstruct the other half of the test set images 

def view(stack, image):  # function to view an image
    ''' View a single image from a chosen stack of images
        
    Keyword arguments: 
        stack -- (str) this should be either 'train', 'test', or 'recon'                 
        
        number -- (int) this is the index number of the image the stack,
                  12,000 images in the training set, 1233 in test set
    '''
    plt.gray()
    
    if stack == 'train':
        plt.imshow(np.hstack((
                            left[image].reshape(64,32), 
                            right[image].reshape(64,32)
                           ))
                  )
                  
    elif stack == 'test':
        plt.imshow(test[image].reshape(64,32))
    
    elif stack == 'recon':
        plt.imshow(np.hstack((
                            test[image].reshape(64,32), 
                            reconstructed[image].reshape(64,32)
                            ))
                  )
                  
def train():
    """Use  this function to view random training images"""
    view('train', random.randint(0,12000))
def recon():
    """Use this function to view random reconstructed images"""
    view('recon', random.randint(0,1233))
    
print 'Use the view, train, and recon functions to view the images'

                           
                          

