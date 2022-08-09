# Monocular Video Odometry Using OpenCV
This is an Python OpenCV based implementation of visual odometery. 

An invaluable resource I used in building the visual odometry system was Avi Singh's blog post:  
http://avisingh599.github.io/vision/monocular-vo/ as well as his C++ implementation found [here](https://github.com/avisingh599/mono-vo).

Datasets that can be used:  
[http://www.cvlibs.net/datasets/kitti/eval_odometry.php](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)  
[http://www.cvlibs.net/datasets/kitti-360/](http://www.cvlibs.net/datasets/kitti-360/)  

# Demo Video
<img src="images/demo6d5ae7a5bb1211ec8095a497b1b39748_cut.gif">
<img src="images/demo69d4b3edbb1611eca9f2a497b1b39748_cut.gif">


# Algorithm
Steps of the algorithm are taken from Avi Singh's blog post mentioned above. 
1. Capture images: I<sup>t</sup> and I<sup>t + 1</sup>  
2. Undistort the captured images, in the demo video I used Kitti 360 dataset, recorded by 2 fisheye cameras so need to undistort according to intrinsic  
3. Use FAST algorithm to detect features in image I<sup>t</sup>. Track these features using an optical flow methodology, remove points that fall out of frame or are not visible in I<sup>t + 1</sup>. Trigger a new detection of points if the number of tracked points falls behind a threshold. Set to 2000 in this implementation. 
4. Apply Nister's 5-point algorithm with RANSAC to find the essential matrix.
5. Estimate R, t from the essential matrix that was computed from Nister's algorithm.
6. Obtain scale information from an external source and concatenate translation vectors t and rotation matrices R.


# Running Program
1. First clone repository
2. In `test.py` change `img_path` and `pose_path` to correct image sequences and pose file paths
3. Ensure focal length and principal point information is correct
4. Adjust Lucas Kanade Parameters as needed
5. Run command `python ./test.py`
