# Vehicle Detection Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


## Goals and steps
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.

## Detail description

[//]: # (Image References)
[image1]: ./readme_img/car_noncar_img.png
[image2]: ./readme_img/spacial_features.png
[image3]: ./readme_img/color_histogram.png
[image4]: ./readme_img/hog_features.png
[image5]: ./readme_img/feature_normalize.png
[image6]: ./readme_img/windows.png
[image7]: ./readme_img/preprocess.png
[video1]: ./project_video.mp4

### data 

### 1. Feature extract 
Extract features based on color space and HOG features.


