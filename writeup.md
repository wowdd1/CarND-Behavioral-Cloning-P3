#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/image1.png "Model Visualization"
[image2]: ./examples/SqueezeNet52.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* p3.ipynb for preprocess csv file
* preprocessed_driver_log.csv processed csv file
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* run1.mp4 for the test result

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

the task is in the simulator environment, not so complex like real data, so i choose squeezenet for do this task

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 94). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 122-128). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 101).

####4. Appropriate training data

I used image augmented for more training data, like shift, flip

###Model Architecture and Training Strategy

####1. Solution Design Approach

i choose squeezenet because the task is not as complex as real task, and the params is small, so the training speed is fast 

####2. Final Model Architecture

The final model architecture (model.py lines 84-102) consisted of a convolution neural network with the following layers and layer sizes

Here is a visualization of the architecture:

![alt text][image2]

![alt text][image1]

####3. Creation of the Training Set & Training Process

I use the udacity training data, and use some augmented method for got more data


After the collection process, I had about 20000 number of data points. I then preprocessed this data by crop the top and bottom of the image


I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 50, I used an adam optimizer so that manually training the learning rate wasn't necessary.
