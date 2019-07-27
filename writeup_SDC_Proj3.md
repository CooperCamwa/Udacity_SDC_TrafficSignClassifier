# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**
The project im simply gategorized in some steps:
    * importing the needes libraries
    * reading and loading the training, evaluating and testing data
    * visualization of 10 images to have a feeling about them
    * Pre-processing the images (is one of the most important part of project)
    * designing the CNN architecture and defining the CNN parameters
    * train the network with feeding the training images
    * evaulating the network with feeding the evaluation images
    * testing the accuracy of network with feeding the test images into traint CNN
    After doing all above mentioned steps, it was the time to test the CNN with some random images which are found in internet:
    * finding 10 random traffic sign images (jpg and png format) from internet and saving them in "new_GTS"
    * preprocess these new 10 images with preprocess function
    * test the CNN with these 10 preprocessed images
    



[//]: # (Image References)

./new_GTS/ "9 images with .jpg and .png format"
./img_examples/original_images.png "original images in RGB"
./img_examples/grayscla_images.png "original images in graysclae"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3), RGB after preprocessing
* The number of unique classes/labels in the data set is 43


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the color is not important for my preprocessing and for my CNN.

Here is an example of a traffic sign image before and after grayscaling.

./img_examples/original_images.png "original images in RGB"
./img_examples/grayscla_images.png "original images in graysclae"



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My CNN looks as follows:

   * reading train data 
   * first convolution step: (input dim = 32, input_depth = 1 (grayscale), output_dim = 28, output_depth = 12) with strides=1, padding='VALID'
   * adding relu function
   * adding maxpool with kernel size=2 and strides = 2, padding='VALID'

   * Second convolution step: (input dim = 14, input_depth = 10, output_dim = 12, output_depth = 25) with strides=1, padding='VALID'
   * adding relu function
   * adding maxpool with kernel size=2 and strides = 2, padding='VALID'
   * flattening
   * adding full-connected layer (input_dim = 625, output_dim = 300)
   * adding full-connected layer (input_dim = 300, output_dim = 100)
   * adding full-connected layer (input_dim = 100, output_dim = number of unique classes = 43)
   

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For train the model, I've used the preprocessed training images with training rate = 0.001 (as smal as possible), epoch number=10 and batch size= 64. I've shuffled the images and lebles and feed the to CNN. Here I've tried to optimized (minimized) the lost (error).

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

   * training set accuracy of 0.994

for validation I've feed the validateion images to my network and checked the accuracy:
    * validation set accuracy of 0.941

for testing, I've feed the test images to CNN and checked the accuracy:
* test set accuracy of 0.936342042746



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 9 German traffic signs that I found on the web:

./new_GTS/
These images are:
    * Vehicles over 3.5 metric tons prohibited
    * Speed limit (30km/h)
    * Yield
    * Dangerous curve to the left
    * Double curve
    * Right-of-way at the next intersection
    * Pedestrians
    * Road work
    * Stop

here all images together:
./img_examples/internet_imgs_original.png
./img_examples/internet_imgs_3232.png

These images are in jpg and png format. Then I've feed them to my preprocess function to have the same preprocessing as train data (it is eesential).
he model was able to correctly guess 3 of the 9 traffic signs, which means accuracy of 33%. not really good result. Some of images (after resizing into 32x32) are not readable with eyes too. and I think would be really difficult for CNN too.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

   * image 1: the model sure with probability of 0.94 [9.37336087e-01,3.89467031e-02,6.19173888e-03,5.79671189e-03,4.87415958e-03] that the image is "Speed Limit 20" ([0,40,17,33,32]), but the image is "Vehicles over 3.5 metric tons prohibited", not correct prediction.
   * image 2: the model (not realy) sure with probability of 0.67 [6.78498924e-01,3.02198082e-01,8.72930139e-03,5.25789196e-03,1.67541765e-03] that the image is "bycicle crossing" and with probability of 0.3 "Speed limit (30km/h)"([29,1,35,37,0]),  the image is "Speed limit (30km/h)", not completely!! correct prediction.
   * image 3: the model sure with probability of 0.94 [9.37336087e-01,3.89467031e-02,6.19173888e-03,5.79671189e-03,4.87415958e-03] that the image is "Yield" ([ 0, 40, 17, 33, 32]), and the image is "Yield", correct prediction.
   * image 4: the model sure with probability of 0.99 [9.99993801e-01,6.25192797e-06,1.33244240e-11,1.98065932e-14,   1.86148950e-16] that the image is "Dangerous curve to the left" ([19,23,3,25,16]), and the image is "Dangerous curve to the left", correct prediction.
   * image 5: the model (not really) sure with probability of 0.76 [7.61180520e-01,2.38819510e-01,4.75663987e-11,1.69812883e-12,1.56489080e-12] that the image is "Children crossing" and with probability of 0.23 "Right-of-way at the next intersection" ([28,11,6,30,20]), but the image is "Double curve", not correct prediction.
   * image 6: the model pretty sure with probability of 1 [1.00000000e+00,2.78936526e-15,1.37700055e-16,5.54559584e-17,1.12640261e-17] that the image is "Right-of-way at the next intersection" ([11,40,30,6,42]), and the image is "Right-of-way at the next intersection", correct prediction.
   * image 7: the model pretty sure with probability of 0.999 [9.99999046e-01,9.60911166e-07,5.48129719e-09,3.75656673e-13,2.33275449e-13] that the image is "Right-of-way at the next intersection" ([11,27,18,30,24]), but the image is "Pedestrians", not correct prediction.
   * image 8: the model completely confused (the highest probability is 0.36!!) [3.66339326e-01,1.82877645e-01,1.11811146e-01,7.31651932e-02,6.95329607e-02] for these images [37, 30, 11, 28, 24] and the image is "Road work", not correct prediction.
   
* image 9: the model (not really) sure with probability of 0.73 [7.29531825e-01,1.62437826e-01,7.84723833e-02,1.48305697e-02,5.70491375e-03] that the image is "Right-of-way at the next intersection" ([11,10,42,13,12]), but the image is "Stop", not correct prediction.


  33% corrected prediction!! not really good. But we have consider that the image quality are not really good.
       
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


