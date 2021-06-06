# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

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
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing frequecy of each class in train, test and validate data .
[image9]: ./write_up_data/train_number_of_samples_on_each_category.png "Training data frequency chart"

[image10]: ./write_up_data/validate_number_of_samples_on_each_category.png "Validation set data frequency chart"

[image11]: ./write_up_data/test_number_of_samples_on_each_category.png "Testing data frequency chart"
![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because color attributes doesnot convey more info about a partucular sign(most sign contain red black and white and sometime yellow).
i.e having 3 channel of color will confuse the network and training will be more time consuming

Then I normalised the image to get zero mean data, so that our omptimiser can work efficiently.
normalisation :- (pixel-128)/128

Here is an example

[image12]: ./write_up_data/image_transformations.png "trnasformed image"

## The difference between the original data set and the augmented data set is the following ... 



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        													| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image(actual imput was 32x32x1 RGB image)							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 									|
| RELU					|																				|
| Max pooling			| 2x2 stride,  outputs 14x14x6 													|
| Convolution 5x5		| 1x1 stride, same padding, outputs 10x10x16 	 								|
| Max pooling	 		| 2x2 stride,  outputs 5x5x16 													|
| Fully connected		| input 400, output 120 														|
| RELU					|																				|
| Fully connected		| input 122, output 84															|
| RELU					|																				|
| Fully connected		| input 84, output 43															|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Optimizer used is AdamOptimizer with learning rate of 0.0001.The epochs used was 48 while the batch size was 128.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.938
* validation set accuracy of  0.951
* test set accuracy of 0.919

I sared with LeNet, first I used color image(32X32X3), accuracy was around 0.915,then I did some preprocessing mentioned above and used(32X32X1) grayscale image.

After that I tuned hyper-parameters to get the desired acuuracy.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

[image10]: ./test_images/1.png "Speed 30kmph"
[image10]: ./test_images/2.png "Bumpy road"
[image10]: ./test_images/3.png "ahead only"
[image10]: ./test_images/4.png "Speed 50kmph"
[image10]: ./test_images/5.png "Go straight or left"
[image10]: ./test_images/6.png "General Cauotion"

The 4th image might be difficult to classify because it has some unwanted background which confuses the network.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                |     Prediction	        					| 
|:---------------------:        |:---------------------------------------------:| 
| Speed Sign 30kmph     		| Speed Sign 30kmph    									| 
| Bumpy road     			    |  Bumpy road   										|
| ahead only					| ahead only											|
| Speed 50kmph	      		    | Speed 60kmph					 						|
| Go Straight or left			| Go Straight or left      								|
| General Caution			    | General Caution     									|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.33%. This compares favorably to the accuracy on the test set of 91.9%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


