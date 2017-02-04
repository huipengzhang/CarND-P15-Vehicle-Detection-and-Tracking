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
* Number of feautures: 384

**Color histogram feature parameters:**

* hist_bins = 12 
* channels: HLS and RGB
* Number of feautures: 72

**HOG feature parameters:**

* orient = 8
* pix_per_cell = 12
* cell_per_block = 2
* channels: Grey scale
* Number of feautures: 512

**Total number of feature:** 968

#### 1.7 Feature Normalization

(The code is is contained in `cell #16`)

The 'StandardScaler()' is used, which removes the mean and scales the features to unit variance. A scaler is training using the training set data and applied to the training and testing set.
Here is an example of the raw and normalized feature.

![alt text][image5]

#### 1.8 Make Training, Testing, and Validation set

The image in the training set is randomly shuffled. The image in the testing set is divided equally into testing set and validation set.

**The number images:**
Training set  :  11732 
Validation set:  3363
Testing set   :  3363

### 2. Define Classifier

#### 2.1 Tuning Classifier Parameters

Random forest algorithm was chosen because it has a good balance of performance and speed.
The algorithm uses the ensemble of decision trees to give a more robust performance.
Classification output probability. A threshold will be set up later to reduce the false positive.

Instead of accuracy, the auroc is used as the performance metric to measure the robustness of the algorithm.

The tuning parameters including  max_features, max_depth,  min_samples_leaf.

#### 2.2 Evaluate the Classifier
The code for this step is contained in cell # in  the notebook.

Smaller tree depth and max features have a better result, at they turn to use more general rules.


n_estimators=100
max_features = 2
min_samples_leaf = 4
max_depth = 25

Training time: about 3 Seconds
Training auroc    = 1.0
Training accuracy = 0.9998
Testing auroc    = 0.9714
Testing accuracy = 0.818
Validation auroc    = 0.9686
Validation accuracy = 0.8156


### 3. vehiche Detection

#### 3.1 Sliding window search
Sliding windows are used to crop small images for vehicle classification.
To Minimize the number of searches, the search area is retrained to area vehicle will appear.
First,  the minimum and maximum size of the window are decided, the intermediate sizes are chosen by interpolation.
The results on different window size are:

Here is an example of search windows with different size

![alt text][image6]

#### 3.2 Extract Features form Windows
First
The pixels in each window are cropped and rescaled to 64x64x3 pixel, which is the same as the training image. 

Then, the classifier determine with the window is a car image or not.


Here is an example shows window of the the detected vhecle for all the test images

![alt text][image7]


False positive around theb fence. 

dark color vhicle. better for more vibrate color

Splite the limitation, it find car, use duplicates detection and tracking to filter out the outplier


### 4. Duplicates Detection.

#### 4.1 Create a Heatmap

Duplicates are multiple detections finding the same car image. To eliminate the duplicate 
a heat-map is build from combining overlapping detections.

False positives are not consistent. It will appear and disappear. 

To filter out false positives as nose. 
I  build 

The individual heat-maps for the above images look like this:

reduce it the value, it cool down if not detected. 

overlay a heat map on the value detection.

Here is an example shows window of show the heatmap box and labels

![alt text][image8]

#### 4.2 Estimate Bounding Box

Set a threshold to filter out as noise. And get disconnected areas. Use label to label the area.as diferent vehicle and draw a bounding box around the area.

Here is an example shows window of the bounding box

![alt text][image9]
 
### 5. Vehicle Tracking

Tracking in a video, tracking pipeline.

I using moving average algorithm with is decrease as 
The old value decrease exponcially, 

I created a car () object to car
The car boject contrains 4 attribute, average_centroid
           width = 0 
           .height = 0
       
      detected = 0.5  is a float value define to measurement how certain the car is detected. i use the moving average to update the value if a car is detected in a frame, value will increase.
      
      after each from the value is deprecate, decrease expensally
      
    code 
    
the the tracking process is discribe as follow:
create two global valuables :
heatmap, Detected_Cars

each frame create a heat map heatmap_new for the window of detected value

the goble valabuble heatmap is updated using moving average. 
 
 heatmap = 0.9*heatmap + 0.1*heatmap_new
 
heatmap_sure thredhold the heatmap filter out sure is indeed a car, and create bounding_boxes 

find centroy and size of bounding box, loop through each centroid to if is to nd nearby car object       

```
        car_found, k = track_car(centroids[n],Detected_Cars) 
```

if car is fund it update the detected cars centroid and bounding box heigh and width, detected value using moving average
```
            # update detected car object
            # update centroid using moving average
            Detected_Cars[k].average_centroid = (int(0.9*Detected_Cars[k].average_centroid[0] + 0.1*centroids[n][0]),
                                    int(0.9*Detected_Cars[k].average_centroid[1] + 0.1*centroids[n][1]))         
            # update bounding box width using moving average
            Detected_Cars[k].width =   math.ceil(0.9*Detected_Cars[k].width + 0.1*box_size[n][0]) # round up
            # update bounding box height using moving average
            Detected_Cars[k].height =  math.ceil(0.9*Detected_Cars[k].height + 0.1*box_size[n][1])
            # update detected value
            Detected_Cars[k].detected = Detected_Cars[k].detected + 0.2
  ```
if not neear by car is found it add a new car object
```
       new_car = car()
            # inicalize the car object using the size 
            # and centroid of the bounding box
            new_car.average_centroid = centroids[n]
            new_car.width =  box_size[n][0]
            new_car.height = box_size[n][1]            
            New_Cars.append(new_car)
```
# combine new_cars to detected cars
    Detected_Cars2 = list(Detected_Cars) # make a copy
    Detected_Cars = New_Cars[:] # add new cars
    if Detected_Cars2: # if is not empty
        for car in Detected_Cars2:
            # if the detected value greater than the threshold add to the list
            # if not discard
            if car.detected > 0.17: 
                # add to the detected cars list
                Detected_Cars.append(car)
            
add new car to the list and if previous detacted car is higher thant threshodl it add it too 
```
if Detected_Cars2: # if is not empty
        for car in Detected_Cars2:
            # if the detected value greater than the threshold add to the list
            # if not discard
            if car.detected > 0.17: 
                # add to the detected cars list
                Detected_Cars.append(car) 
   ```
   
  last it   depreciate old car values, so if it no longer detacted the value fade away
  
  ```
    for car in Detected_Cars:
        car.detected = car.detected*0.8 # depreciate old value
```

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



![alt text][image10]




 ### 6. Video pipeline
The pipeline works as follows, process the video frame by frame, sliding window to search for car image, create heat map for group overlapping windows, track the detected blog. 

Create a pipline to detect cars in a video stream Visualization:

* Detected windows: Blue boxes
* Heatmap: Green area
* Bounding boxes of cars: Red boxes

The produce many False Positives, , especially around theb fences on the left side. 
It possilble the fences have vertical lines which can be confusing to a cars image.

The white vechile, and dark vehicle can be detected. 
The threshoding the heat map reduce, the among, hover, it also take a second to confornt it is a car. of lose track.


Here's a [link to my video result](https://www.youtube.com/watch?v=Djb4ydFqc7U)

---
## Discussion

### 1. Problems faced in your implementation of this project

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


### 2.Likely failure and possible improvement

From the final video results, the classifiers still produce many False Positives, especially around theb fences on the left side. 
It possilble the fences have vertical lines which can be confusing to a cars image.

The classifer achieve 84% of the accuracy.

Although the seting threshodl on heatmap can reduce, number of false positives

Deeplearning can improve in the accruacy.



One possible false can be caused by 
False positive around theb fence. 

dark color vhicle. better for more vibrate color
The classifer 
From the final video results, in the shadows area. False positive. The robustness of the classifier can still be increase. More specifically, it can be better distinguish, shadow and cars. 
The track of two vehicle move very close. The algorithm will just group them as one big vehicle. Instead of two.

Moving average algorithm introduce delay, the box lag belind

