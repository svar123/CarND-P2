
# **Traffic Sign Recognition** 

## Writeup Template
---

### **Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./newimages/color.jpg "Visualization"
[image2]: ./newimages/trainbar.jpg "Train bar plot"
[image3]: ./newimages/validbar.jpg "Validation bar plot"
[image4]: ./newimages/testbar.jpg  "Test bar plot"
[image5]: ./newimages/grayimage.jpg "Gray Image"
[image6]: ./newimages/loss.jpg "Loss Plot" 
[image7]: ./newimages/accuracy.jpg "Accuracy Plot"
[image8]: ./newimages/newcolor.jpg "New Color"
[image9]: ./newimages/newgray.jpg "New Gray"
[image10]:./newimages/prob0.jpg "First"
[image11]:./newimages/prob1.jpg "Second"
[image12]:./newimages/prob2.jpg "Third"
[image13]:./newimages/prob3.jpg "Fourth"
[image14]:./newimages/prob4.jpg "Fifth"
[image15]:./newimages/prob5.jpg "Sixth"
[image16]:./newimages/prob6.jpg "Seventh"
[image17]:./newimages/prob7.jpg "Eighth"
[image18]:./newimages/prob8.jpg "Ninth"
[image19]:./newimages/prob9.jpg "Tenth"
[image20]:./newimages/feature_map.jpg "Feature"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/svar123/CarND-P2/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set = 34799
* The size of the validation set = 4410
* The size of test set = 12630
* The shape of a traffic sign image = (32,32,3)
* The number of unique classes/labels in the data set = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The following is a representation of ten random input images.

![alt text][image1]

The following bar plots shows the distribution of classes in the training set, validation set, and test set.

![alt text][image2]

Max value =  2010

Min value =  180

Mean value =  809.27

![alt text][image3]

Max value =  240

Min value =  30

Mean value =  102.55

![alt text][image4]

Max value =  750

Min value =  60

Mean value =  293.72


The distributions of the classes over the three sets shows similarity with frequency of classes 1, 2, 13 being high and classes 0, 19, 37 being low.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because this would reduce the complexity of the model and also it is easier for the algorithm to detect edges in grayscale. To do this, I used the image processing common formula :
Gray = 0.299 Red + 0.587 Green + 0.114 Blue

The image data was then normalized as follows for training, validation and testing:

X_train = X_train/255.0 - 0.5 

X_valid = X_valid/255.0 - 0.5

X_test = X_test/255.0 - 0.5

This made sure all values are betweeen -0.5 and 0.5. Normalization will make converging faster.

Here are examples of traffic sign images after grayscaling and normalizing.
![alt text][image5]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 		    		|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 		    		|
| Flatten               | 5x5x16 -> 400                                 |
| Fully connected		| 400 -> 120        							|
| RELU				    |         								        |
| Dropout				| 												|
| Fully connected		| 120 -> 84										|
| RELU				    |         								        |
| Dropout				|												|
| Fully connected		| 84 -> 43										| 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used LeNet architecture as a baseline. I used the default values for batch size(128), learning rate(0.001), mean(0) and sigma(0.1). The number of epochs was set to 25.

I used softmax_cross_entropy_with_logits function to measure the probability, and AdamOptimizer for the training operation.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.
 

My final model results were:
* training set accuracy of 99.8%
* validation set accuracy of 94% 
* test set accuracy of 93.4%

The following is a loss plot for training and validation data.
![alt text][image6]
And this is the accuracy plot for the training and validation data.
![alt text][image7]

I started with LeNet architecture as a baseline with no changes. The results showed significant difference between training accuracy(99.2%) and validation accuracy (89%)due to overfitting. Then I tried adding dropout layer before the last fully-connected layer and saw improvement. Adding a second dropout layer after the first fully-connected layer gave the above accuracy results proving the model is working well.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web. I downloaded the files as ppm and converted it to jpg format.
![alt text][image8]

The images were converted to gray-scale and normalized.
![alt text][image9]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Vehicles over 3.5     | Vehicles over 3.5   							| 
| 30km/hr     			| 30km/hr										|
| Keep right			| Keep right									|
| Turn right	      	| Turn right					 				|
| Right of way			| Right of way     						    	|
| Keep right			| Keep right									|
| Caution   	      	| Caution    					 				|
| Priority road			| Priority road     					    	|
| Road work   	      	| Road work    					 				|
| Ahead only			| Ahead only     			    		    	|


The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.4%. The model performed well on all the new images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 41st cell of the Ipython notebook.

Here are the ten images:
![alt text][image8]


For the first image, the model is sure that this is a sign Vehicles over 3.5 metric tons prohibited(prob of 1.00000000e+00 ), and the image does contain this sign. The top five softmax probabilities (1.0,5.3e-9,7.6e-11,7.4e-11,1.7e-11)with the sign names are shown in the bar plot below.  

![alt text][image10]

For the second image, the model is sure that this is a Speed limit(30km/h) sign (prob of 1.00000000e+00 ), and the image does contain this sign. The top five softmax probabilities (1.0,8.5e-09,2.2e-09,6.6e-10,2.8e-11)with the sign names are shown in the bar plot below. 

![alt text][image11]
For the third image, the model is sure that this is a sign Keep right (prob of 1.00000000e+00 ), and the image does contain this sign. The top five softmax probabilities (1.0,3.04e-20,1.2e-24,2.4e-25,2.2e-25)with the sign names are shown in the bar plot below. 

![alt text][image12]
For the fourth image, the model is sure that this is a sign Turn right ahead(prob of 1.00000000e+00 ), and the image does contain this sign. The top five softmax probabilities (1.0,3.04e-09,2.3e-09,2.6e-10,1.6e-11)with the sign names are shown in the bar plot below. 

![alt text][image13]
For the fifth image, the model is sure that this is a sign Right-of-way (prob of 1.00000000e+00 ), and the image does contain this sign. The top five softmax probabilities (9.99e-01,2.75e-06,1.9e-08,9.4e-09,3.43e-09)with the sign names are shown in the bar plot below. 

![alt text][image14]
For the sixth image, the model is sure that this is a sign Keep right(prob of 1.00000000e+00 ), and the image does contain this sign. The top five softmax probabilities (1.0,2.1e-18,1.6e-22,1.3e-22,3.6e-23)with the sign names are shown in the bar plot below. 

![alt text][image15]
For the seventh image, the model is sure that this is a sign General caution(prob of 1.00000000e+00 ), and the image does contain this sign. The top five softmax probabilities (1,8.2e-13,2.8e-14,8.09e-21,2.4e-21)with the sign names are shown in the bar plot below. 

![alt text][image16]
For the eighth image, the model is sure that this is a sign Priority road(prob of 1.00000000e+00 ), and the image does contain this sign. The top five softmax probabilities (1,2.8e-09,9.7e-12,2.6e-14,9.6e-15)with the sign names are shown in the bar plot below. 

![alt text][image17]
For the ninth image, the model is sure that this is a sign Road work(prob of 1.00000000e+00 ), and the image does contain this sign. The top five softmax probabilities (1,1.1e-17,8.3e-19,5.8e-19,2.1e-20)with the sign names are shown in the bar plot below. 

![alt text][image18]
For the tenth image, the model is sure that this is a sign Ahead only(prob of 1.00000000e+00 ), and the image does contain this sign. The top five softmax probabilities (1.0,1.49e-09,8.4e-10,3.9e-12,2.9e-12)with the sign names are shown in the bar plot below. 

![alt text][image19]




### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image20]
The feature maps  have detected the circular edges of the sign.


```python

```
