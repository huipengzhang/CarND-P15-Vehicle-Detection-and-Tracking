# Vehicle Detection Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


## Goals and steps
The goals / steps of this project are the following:

* Training a classifier to detect vehicle images using color space feature, color histogram features, and HOG features.

* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.

* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.

* Estimate a bounding box for vehicles detected.

Here's a [link to my video result](https://www.youtube.com/watch?v=Djb4ydFqc7U)

Links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.

---

## Detail description

The code is contained in the IPython notebook `Object_Detection.ipynb`. I will refer to code location
using the cell#.

[//]: # (Image References)
[image1]: ./readme_img/car_noncar.png
[image2]: ./readme_img/spatial.png
[image3]: ./readme_img/color_hist.png
[image4]: ./readme_img/hog.png
[image5]: ./readme_img/normalize.png
[image6]: ./readme_img/windows.png
[image7]: ./readme_img/detect.png
[image8]: ./readme_img/heatmap.png
[image9]: ./readme_img/bbox.png


### 0. Load Data 

**Load Image Directory**
Reading in all the directory of car and non-car images. ('cell #2' )
The data is from cropping from video stream, the image from the same fold can be very similar. If just randomly split train and test it will cause the test data leak into the training. Set. 
So I choose my train and testing data from different folder
The training data from folder:
The testing and validation data later splitted. 


* Training set

cars: GTI_far, GTI_left, GTI_right, GTI_MiddleClose 

non-cars: Extra

* Test set: cars: 

KITTI_extracted 

non-cars: GTI

The testing and validation data later splitted. 

**Data summary**
Next, I printed out some basic information of the data set such as number of image in each class, image size, and data type '(cell #4)'.
I choose a roughly balanced data set contains 5966 cars image and 5766 non-car images.
Here is an example of one of each of the vehicle and non-vehicle classes:

![alt text][image1]


### 1. Define Features

The next step is to define features for the vehicle classification.Three types of features are used: color space feature, color histogram features, and HOG features.

**Convert Image Datatype**

The images in the training data set are of the jpeg format, with float data values range from 0-1. The test images are of the png format, range from 0-255. To be consistent with the images type in the later process. I first convert the training image data type to int type with value from 0 to 255 (cell # 5)

**Color Space Feature**
Color space feature uses the raw pixel values of the images and flattens them into a vector. To reduce the size of the image, I performed spatial binning on an image by resizing the image to the lower resolution.

To reduce the number of features, only the saturation channel in the HLS color space is used, based on the assumption that the saturation channel would be a good representation of the image, because the cars are more likely to have a more prominent appearance.

Here is an example of an image in S_channel and the value of color space features.
![alt text][image2]



(cell #6-7)

**Color Histogram Features**
The code for this step is contained in cell # in  the notebook.

Color Histogram feature is more robust to the different the appearance of the car.  The Color Histogram remove the structural relation and allow more flexibility to the variance of the image.
Binning is performed to the histogram of each channel. Both the RGB and HLS channels are used. 

![alt text][image3]

** Histogram of Oriented Gradients (HOG) **
Gradient features is also used to capture the signature for a shape. However use the gradient feature directly is sensitive.  Histogram of gradient orientation allows variation  between the shape. The HOG is on the grey scale image. 

The code for this step is contained in cell # in  the notebook.

![alt text][image4]

### 2. Define Classifier
The code for this step is contained in cell # in  the notebook.

**Tuning Classifier Parameters**
Random forest algorithm was chosen because it has a good balance of performance and speed.
The algorithm uses the ensemble of decision trees to give a more robust performance.
Classification output probability. A threshold will be set up later to reduce the false positive.


** Training of the classifer**
The code for this step is contained in cell # in  the notebook.

The steps of training the classifier are as follows:
Load cars and non-car images through the directory list. 

Standardize features by removing the mean and scaling to unit variance


Use standard scaler is used to normalize the features so that they have similar numerical values.

Shuffled the training data set.

Divide the testing data set into 50% test set and 50% validation set. 

Instead of accuracy, the auroc is used as the performance metric to measure the robustness of the algorithm.

The tuning parameters including  max_features, max_depth,  min_samples_leaf.

The parameters are searched similar to grid search. I found the best parameters are max_feature = 3, max_depth = 5, min_sampe_leaf = 5.  Smaller tree depth and max features have a better result, at they turn to use more general rules.


### 3. vehiche detection

** Sliding window search **
Sliding windows are used to crop small images for vehicle classification.
To Minimize the number of searches, the search area is retrained to area vehicle will appear.
First,  the minimum and maximum size of the window are decided, the intermediate sizes are chosen by interpolation.
The results on different window size are:

The pixels in each window are cropped and rescaled to 64x64x3 pixel, which is the same as the training image. 

Then, the classifier determine with the window is a car image or not.

Test results on the test images are:

5. Duplicates Detection.

Duplicates are multiple detections finding the same car image. To eliminate the duplicate 
a heat-map is build from combining overlapping detections.

False positives are not consistent. It will appear and disappear. 

To filter out false positives as nose. 
I  build 

The individual heat-maps for the above images look like this:

reduce it the value, it cool down if not detected. 

overlay a heat map on the value detection.


Set a threshold to filter out as noise. And get disconnected areas. Use label to label the area.as diferent vehicle and draw a bounding box around the area.


 
### 4. tracking.
Tracking in a video, tracking pipeline.

I using moving average algorithm with is decrease as 
The old value decrease exponcially, 

I created a car () object to car
The car boject contrains 3 attribute, 

Heatmap, 
Update the heat map using moving average algorithm 
Threshod up

Position in one frame,  
Record the position of window
Use tracking to filter out false positive


The the new bounding boxes it calculate the distance to see find if there is a nearby car object. If the distance is within a threshold it. Update the car object. 

It not car object is found, it probability a new car. 
The algorithm create a new car object.

Display car. It only distance car that has detected value higher than threshold. 

This ensure consistency, filter false positive.
 

Defined car object. Centroid, width, height of the bounding box, and detected.

Moving average algorithm is used to update the value. The advantage in average of a queue of the recent value. It doesn’t need to store previous value


 In practice, you will want to integrate a heat map over several frames of video, such that areas of multiple detections get "hot", while transient false positives stay "cool". You can then simply threshold your heatmap to remove false positives.


 ### 4. Video pipeline
The pipeline works as follows, process the video frame by frame, sliding window to search for car image, create heat map for group overlapping windows, track the detected blog. 
The detected windows are shown in blue boxes
The heatmat over lapped.
The bounding bound in red
Here's a link to my video result

---
## Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

The first issue I faced is that I had good training results on the vehicle and non-vehicle training dataset, but poor result on test images from the video stream.

I originally use the "train_test_split" function on the images from the same folder and get very high test accuracy (99%) without any tuning of the parameter. It indicates overfitting.

The problem is that the images from the same folder are very similar because they often are cropped from the same video stream.

So  I used the images from a different folder for the validation and test set. And I found my the linear SVM classifier can only achieve an accuracy of 58%.

And therefore, I decided to switch to a more sophisticated classifier with good running time. And I choose random forests.  It gets an initial accuracy of  70% and after tuning the accuracy increase to 84 % on the test set.

To measure the robustness of the algorithm. I changed the performance metric to “auroc”. And tune the parameters such as: max_features, max_depth,  min_samples_leaf .

The best set of parameters I found are: max_feature = 3, max_depth = 5, min_sampe_leaf = 5.  
The classifier have better performance using “shallow” trees with less maximum features which generate more general result.

After the parameter tuning the classifier, the classification improvement a lot, but it still have some miss classifies.

@The second problem is to reduce the false positives, which mean classifier identify cars which is not there. I took a combination of 1. increasing the robustness of the classifier. 2.  Filter the mistake by tracking. 
In the first step, I change the classifier to produce a probability instead of a binary classifier. Then I can increase the threshold, for example 50% sure, need to be 60% sure. However, I found the classifier have very narrow margin.e.g. Just increase from 50% to 55%, it will miss some true positive.

So I try the second approach.


### Fail
From the final video results, in the shadows area. False positive. The robustness of the classifier can still be increase. More specifically, it can be better distinguish, shadow and cars. 
The track of two vehicle move very close. The algorithm will just group them as one big vehicle. Instead of two.

Moving average algorithm introduce delay, the box lag belind

