## Extended Kalman Filter and Sensor Fusion
---

**Bicycle Tracking Project**

The goals / steps of this project are the following:

* Implement an Extended Kalman Filter to fuse data from radar and laser sensors to track the state of a cyclist.

[//]: # "Image References"
[image1]: ./images/rmse.png
## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
####1. Code must compile without errors with `cmake` and `make`.

Done.  Just try it.

####2. Tracking Accuracy.

RMSE tracking errors are [X : 0.0977, Y : 0.0854, VX : 0.4406, VZ : 0.4608]

![alt text][image1]

####3. My Sensor Fusion algorithm follows the general processing flow as taught in the online lessons.

* The Kalman filter algorithm is implemented in [./src/kalman_filter.cpp](./src/kalman_filter.cpp).
* Note that I use a slightly different formulation for the covariance measurement update to help preserve symmetric positive definiteness of the state covariance (perhaps not important for this particular project). 

####4. The Kalman Filter algorithm handles the first measurements appropriately.

* Very basic state initialization is implemented.  If we are willing to allow initialization after the first $n>1$ measurements, then we can obtain better initialization of the full state.

####5. The Kalman Filter algorithm first predicts then updates.

* That's what it does.

####6. The Kalman Filter can handle radar and lidar measurements.

* Yup.

#### 7. The code is efficient

* For the most part, I followed the provided outline.  However, I did use a more _functional programming_ approach to the implementation of the Kalman Filter itself.  Notice that the Kalman filter defined  in kalman_filter.cpp is not an object and has no state.
* I prefer this approach because otherwise we have to know the "secret handshakes" to use the kalman filter object.  "Aaah, you set R_radar and R_laser, but did you set Q?".  Nothing is going to warn if you don't, and no one will tell you if somebody or something else changes those values under your feet before you get to make the actual call of the desired update function.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  

* As always, the EKF works remarkably well, far better than it should in theory.  Other approaches may improve on the EKF slightly, but it is hard to justify the extra computational overhead.
* I did have a bug in my code (an extra factor of 2 on my radar measurement function).  Performance was marginal (about double the desired error).  Key to discovering the error was running with only laser, and then only radar.  Laser worked much better just by itself.  This helped confirm that the underlying Kalman filtering algorithm was correctly coded, and that the problem lay in the radar processing, which quickly lead to the culprit.
* The interacting multiple model filter could be used in this example.  For example, we could use three models: _left turn_, _right turn_, and _straight_.  Learning which model is currently in play would allow for more accurate update prediction.

