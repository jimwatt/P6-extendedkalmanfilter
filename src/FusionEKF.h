#ifndef FusionEKF_H_
#define FusionEKF_H_

#include "measurement_package.h"
#include "Eigen/Dense"
#include "kalman_filter.h"
#include "tools.h"

class FusionEKF {
public:

  FusionEKF();

  // update the state with the given measurement
  void ProcessMeasurement(const MeasurementPackage &measurement_pack);

  // state (mean and covariance)
  Eigen::VectorXd x_;
  Eigen::MatrixXd P_;

private:
  // check whether the tracking toolbox was initialized or not (first measurement)
  bool is_initialized_;

  // previous timestamp
  long long previous_timestamp_;

  // tool object used to compute Jacobian and RMSE
  Tools tools;

  // Measurement noise
  Eigen::MatrixXd R_laser_;
  Eigen::MatrixXd R_radar_;
  
  // Laser measurement function
  Eigen::MatrixXd H_laser_;

  // Process noise
  double s2ax_;
  double s2ay_;
};

#endif /* FusionEKF_H_ */
