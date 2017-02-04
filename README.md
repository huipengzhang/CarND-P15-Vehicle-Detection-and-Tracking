# Vehicle Detection Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


## Goals and steps
The goals/steps of this project are the following:

1. Define Features: define features for the vehicle classification including color space feature, color histogram features, and HOG features.

2. Define Classifier: train and fine tune a random forests classifier  to detect vehicle.

3. Vehicle Detection: implement a sliding-window technique and use the classifier to determine whether the image contains vehicles

4. Duplicates Removal: create a heatmap to removal duplicates  (multiple detections  of the same car) and outliers.

5. Vehicle Tracking: tracking and estimate a bounding box for vehicles detected.

6. Video Pipeline: run the pipeline on a video stream and detect vehicles frame by frame


Here's a [link to my video result](https://www.youtube.com/watch?v=Djb4ydFqc7U)

Links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.

---

## Detail description

The code is contained in the IPython notebook `Vehicle_Detection.ipynb`. I will refer to code location
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
[image10]: ./readme_img/eqn.png


### 0. Load Data 

**0.1 Load Image Directory**

(The code is is contained in `cell #2`)

Reading the directory of the car and non-car images. Since the data is from cropping from the video stream, the image from the same fold can be very similar. If just randomly split train and test, it will cause the test data to leak into the training. So I choose the train and testing data from different folders, shown as follows:

**Training set**

* cars: GTI_far, GTI_left, GTI_right, GTI_MiddleClose 
* non-cars: Extra

**Test set**

* cars: KITTI_extracted 
* non-cars: GTI

#### 0.2 Data Summary

(The code is contained in `cell #4`)

Next, I printed out some basic information of the data set such as the number of the image in each class, image size, and data type.
I choose a roughly balanced data set, which contains 5966 cars image and 5766 non-car images. Here is an example of one of each of the vehicle and non-vehicle classes:

![alt text][image1]


### 1. Define Features

The next step is to define features for the vehicle classification.Three types of features are used: 

* Spatial feature
* Color histogram features 
* HOG features.

#### 1.1 Convert Image Datatype

(The code is contained in `cell #6`)

The images in the training data set are of the jpeg format, with float data values range from 0-1. The test images are of the png format, with int data values range from 0-255. To be consistent with the images type in the later process. I first convert the training image data type to int type with value from 0 to 255.

#### 1.2 Spatial Feature

(The code is contained in `cell #7`)

The spatial feature uses the raw pixel values of the images and flattens them into a vector.  I performed spatial binning on an image by resizing the image to the lower resolution. To reduce the number of features, only the saturation channel in the HLS color space is used, based on the assumption that the saturation channel would be a good representation of the image, because the cars are more likely to have a more prominent appearance. Here is an example of an image in Saturation Channel and the value of the Spatial features.

![alt text][image2]

#### 1.3 Color Histogram Features

(The code is contained in `cell #9`)

Color Histogram feature is more robust to the different the appearance of the car.  The Color Histogram remove the structural relation and allow more flexibility to the variance of the image. Binning is performed to the histogram of each channel. Both the RGB and HLS channels are used. 

Here is an example of the color histogram feature in GRB and HLS color space.

![alt text][image3]

#### 1.4 Histogram of Oriented Gradients (HOG)

(The code is is contained in `cell #11`)

The Histogram of Gradient Orientation (HOG) is also used to capture the signature for a shape and allows variation. The HOG is performed on the gray scale image. Here is an example of the HOG feature.

![alt text][image4]

#### 1.5 Extract Features from the Training Images

(The code is is contained in `cell #13`)

Create a pipline to extract feature form the dataset.

This step creates a pipeline to extract features from the dataset. The feature extraction parameters need to balance the performance and running time. After trial and error, I found the performance doesn't increase much after 1000 features. To keep algorithm run in real times, I keep the number of features around 1000. The feature extraction parameters are as follows:

**Sptial feature parameters:**

* spatial = 8 
* channels:  HLS and RGB
* number of feautures: 384

**Color histogram feature parameters:**

* hist_bins = 12 
* channels: HLS and RGB
* number of feautures: 72

**HOG feature parameters:**

* orient = 8
* pix_per_cell = 12
* cell_per_block = 2
* channels: Grey scale
* number of feautures: 512

**Total number of feature:** 968

#### 1.6 Feature Normalization

(The code is is contained in `cell #16`)

The 'StandardScaler()' is used, which removes the mean and scales the features to unit variance. A scaler is training using the training set data and applied to the training and testing set.
Here is an example of the raw and normalized feature.

![alt text][image5]

#### 1.7 Make Training, Testing, and Validation set

The image in the training set is randomly shuffled. The image in the testing set is divided equally into testing set and validation set.

**The number images:**
Training set  :  11732 
Validation set:  3363
Testing set   :  3363

### 2. Define Classifier

#### 2.1 Tuning Classifier Parameters

(The code is is contained in `cell #23`)

Random forest algorithm was chosen because it has a good balance of performance and speed. The algorithm uses the ensemble of decision trees to give a more robust performance. Classification output probability and a threshold will be set up later to reduce the false positive. 

The tuning parameters including  max_features, max_depth,  min_samples_leaf. `auroc`is used as the performance metric to measure the robustness of the algorithm. 

A grid search is conducted to optimize the parameters. smaller max_features have better performance because the classifier tends to find more general rules. The final set of parameters are as follows

n_estimators = 100
max_features = 2
min_samples_leaf = 4
max_depth = 25

#### 2.2 Evaluate the Classifier

(The code is is contained in `cell #24`)

The performance of the classifier on training testing, and validation set is shown as follows:

Training time: about 3 Seconds
Training auroc    = 1.0
Training accuracy = 0.9998
Testing auroc    = 0.9714
Testing accuracy = 0.818
Validation auroc    = 0.9686
Validation accuracy = 0.8156


### 3. Vehiche Detection

Using the classifier on sliding windows to detect whether an image contain cars.

#### 3.1 Sliding Window

(The code is is contained in `cell #28`)

Sliding windows are used to crop small images for vehicle classification.To minimize the number of searches, the search area is retained to the area where vehicles are likely to appear.First,  the minimum and maximum size of the window are decided. Then, the intermediate sizes are chosen by interpolation. Here is an example of search windows with different size.

![alt text][image6]


#### 3.2 Extract Features form Windows

(The code is is contained in `cell #32`)

First, the pixels in each window are cropped and rescaled to 64x64x3, which is the same as the training image. Then, the classifier determine with the window is a car image or not. Here is an example shows window of the detected vehicle for all the test images:

![alt text][image7]

The classifier gets many False Positives on fences on the left side. It's possible the fences have vertical lines which can be confusing to car images. I also miss the car with darker. Proablby because the color of the car is not very prominent. But, overall the classifier does a good job in finding cars images



### 4. Duplicates Removal

Create a heatmap and apply threshold to removal duplicates (multiple detections of the same car)

#### 4.1 Create a Heatmap

(The code is is contained in `cell #39`)

To eliminate the duplicate, first, a heatmap is built from combining the windows which have car detected. Then a threshold is added to filter out the False Positives. Since False Positives are not consistent. It will be more likely to appear and disappear. The tracking of a vehicle is done across many frames, which will be described in section 5. After the heatmap is thresholded. Use 'label' to find all the disconnected areas. Here is an example shows the heatmap box and labeled areas.

![alt text][image8]

#### 4.2 Estimate Bounding Box

(The code is is contained in `cell #43`)

A bounding box is estimated by drawing a rectangular around the labeled area. Here is an example shows window of the bounding box:

![alt text][image9]
 
 
### 5. Vehicle Tracking

(The code is is contained in `cell #45`)

I created a `car` object to track the detected cars, which contrains 4 attribute, `average_centroid`, `width`, `heigh`, `detected`. 
The `detected` is a float value to measurement how certain detection is. If the car object is detected in a frame, the value will increases if not the value will decrease. 

I created global variable to tracked for tracking: `heatmap` and `Detected_Cars`.  if an area has vheicle detected `heatmap` will heat up, if not the vehicle is detect will 'cool down'. 'Detected_Cars' is a list of previous detacted car object, if `detected` is below a threldshold, it will be removed from the list. 


The the tracking process is discribe as follow: (The code is is contained in `cell #50`)

In each frame, a new heat map `heatmap_new` is created for the window that contains car images. Then the goble valabuble  `heatmap` is updated using the moving average method. 

```
 heatmap = 0.9*heatmap + 0.1*heatmap_new
```

The moving average method is shown as follows:

![alt text][image10]

It is a weighted average of the previous average and the new value.The advantage of this method is that it doesn't need to store all the previous values, and only keep the value of the previous average. The old value decreases exponentially and fades out. The `heatmap_sure` threshold the heatmap to show result with more certainty and create `bounding_boxes`

After finding the bounding boxes. I calculate the distance between the centroid of the bounding box to the centroid of previously detected cars to see find if there is a nearby car object.

```
        car_found, k = track_car(centroids[n],Detected_Cars) 
```

If the distance is within a threshold. It updates the previous car object, with the new centroid, width, and height using the moving average method.

```
            Detected_Cars[k].average_centroid = (int(0.9*Detected_Cars[k].average_centroid[0] + 0.1*centroids[n][0]),
                                    int(0.9*Detected_Cars[k].average_centroid[1] + 0.1*centroids[n][1]))         
            Detected_Cars[k].width =   math.ceil(0.9*Detected_Cars[k].width + 0.1*box_size[n][0]) # round up
            Detected_Cars[k].height =  math.ceil(0.9*Detected_Cars[k].height + 0.1*box_size[n][1])
            Detected_Cars[k].detected = Detected_Cars[k].detected + 0.2
```
if no car is found nearby, I create a new car object

```
            new_car = car()  
            new_car.average_centroid = centroids[n]
            new_car.width =  box_size[n][0]
            new_car.height = box_size[n][1]            
            New_Cars.append(new_car)
```
Then I combine `new_cars` to `Detected_Cars` and loop through the previous `Detected_Cars`.  If the `detected` value is greater than the threshold, it is kept and if not discarded.

```
    if Detected_Cars2: # if is not empty
        for car in Detected_Cars2:
            if car.detected > 0.17: 
                Detected_Cars.append(car)
```

At last, I depreciate the `detected` values, so if a car is no longer detected the value decreases exponentially.

```
    for car in Detected_Cars:
        car.detected = car.detected*0.8 # depreciate old value
```

 ### 6. Video Pipeline
 
The pipeline performs the vehicle detection and tracking on each frame.The results are visualized and overlaid on the original images:

* Detected windows: Blue boxes
* Heatmap: Green area
* Bounding boxes of cars: Red boxes

Here's a [link to my video result](https://www.youtube.com/watch?v=Djb4ydFqc7U)
From the result, we can see the classifier gets many False Positives, especially on fences on the left side. It possible the fences have vertical lines which can be confusing to car images. Some of them even contain cars form the opposite direction. However, they are filter outed, the red box shows the confirmed cars object. The position of the box is lagged behind, due to the delay introduce by the moving average method.



---
## Discussion

### 1. Problems faced in the project

The first issue I faced is that I got good training results on the training dataset, but the poor result on test images from the video stream.

I originally use the `train_test_split` function on the images from the same folder and get very high test accuracy (99%) without any tuning of the parameter. It indicates overfitting. The problem is that the images from the same folder are very similar because they are often cropped from the same video stream. So  I used the images from different folders for the validation and test set. And I found my the linear SVM classifier can only achieve an accuracy of 58%. So, I decided to switch to a more sophisticated classifier with good running time. And I choose random forests.  It gets an initial accuracy of  70% and after tuning the accuracy increase to 84 % on the test set. After the parameter tuning the classifier, the classification improvement a lot, but it still has some miss classifies. 

The second problem is to reduce the false positives, which mean classifier identify cars which are not there. First, I try to increase the robustness of the classifier by change the threshold  probability. However, I found the classifier have very narrow margin.e.g. Just increase from 50% to 55%, it will miss some true positive. So, I try the second approach by filtering and tracking.  I use moving average method to update a heatmap, a set threshold to filter out results that are not very certain. This method eliminates a lot of false positives because the positives are not persistent, they appear and quickly disappear.

### 2.Possible failures and improvements

From the final video results, the classifiers still get many False Positives, especially around the fences on the left side. It possible the fences have vertical lines which can be confusing to car images. By setting a threshold on heatmap, I was able to reduce many of the False Positive. However, sometimes the missing the at some frame. The moving average method used to update the position of the bounding box also introduce some delay, as we can see the bounding box is "lagged" behind the vehicle.

One way to the classification result is to use Deep-learning, the current random forests classifier can achieve an accuracy of 84%. It performs worse on fence and shadow. By using Deep-learning, we can expect a much higher accuracy.Also, the threshold and moving average parameter can also be better tuned to filter more the False Positives without missing many True Positives or introduce to much delay.




