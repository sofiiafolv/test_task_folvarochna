# Task 2. Computer vision. Sentinel-2 image matching
## Solution explanation
Solution is divided in three steps:
* Image preprocessing
* Feature extraction
* Template matching

### Image preprocessing
To preprocess images, I grayscaled them and used resizing with padding and keeping aspect ratio, so all photos are in the same format with size 1024x1024. In this way, some information is lost, but it is easier to work with them.

### Feature extraction
I used the classical CV approach for this task. I used SIFT to extract keypoints and descriptors.

Keypoints are points of interest which define what is interesting or what stands out in the image. Despite any changes in the image, keypoints remain the same.

Descriptors are concerned with both scale and the orientation of the keypoint, so they help us to match keypoints in different images.

### Template matching
FLANN stands for Fast Library for Approximate Nearest Neighbors. It works faster than BF matching, but Brute-Force Matcher could also be used for this task. FLANN would be more useful if we keep the image size as it is. BF matcher matches the descriptors of one set with the closest description of another set. After that I used standard distance for matching ratio 0.7 to remove matches between descriptors that are not close. FLANN uses a collection of algorithms optimized for fast nearest neighbor search in large datasets and for high dimensional features. (https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html?ref=blog.roboflow.com)

## Project setup
All project setup is in "demo.ipynb":

Install necessary packages:
```!pip install -r requirements.txt```

Running algorithm with adding image pathes, which we want to match as arguments.
```!python image_matching_algorithm.py --first_image_path /home/sofiiafolv/UCU/test_task_folvarochna/Task_2/dataset/T36UXA_20180726T084009_TCI.jp2 --second_image_path /home/sofiiafolv/UCU/test_task_folvarochna/Task_2/dataset/T36UXA_20180731T083601_TCI.jp2```

Result is saved in file named "matched_image.png"


