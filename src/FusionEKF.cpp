#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

// Function to wrap an angle from -pi to pi
double ModPosNegPi(const double xraw) {
    const double x = xraw + M_PI;
    const double y = 2.0*M_PI;
    const double m = x - y * floor(x/y);
    if (m>=y) 
        return 0;
    if (m<0) {
        if (y+m == y)
            return 0;
        else
            return y + m; 
    }
    return m - M_PI;
}

// Create the process noise covariance for elapsed time and acceleration noise
MatrixXd makeProcessNoiseCovariance(const double dt, const double s2ax, const double s2ay) {
  MatrixXd Q = MatrixXd::Zero(4,4);
  const double dt2 = dt*dt;
  const double dt3 = dt2*dt;
  const double dt4 = dt3*dt;
  Q(0,0) = 0.25*dt4*s2ax;
  Q(1,1) = 0.25*dt4*s2ay;
  Q(2,2) = dt2*s2ax;
  Q(3,3) = dt2*s2ay;
  Q(2,0) = Q(0,2) = 0.5*dt3*s2ax;
  Q(3,1) = Q(1,3) = 0.5*dt3*s2ay;
  return Q;
}

// Measurement residul for radar
VectorXd zminushx_radar(const VectorXd& z, const VectorXd& x) {
  const double r = x.segment(0,2).norm();
  const double phi = atan2(x[1],x[0]);
  const double rdot = (x[0]*x[2] + x[1]*x[3]) / (r + 1e-8);
  VectorXd zmhx = VectorXd::Zero(3);
  zmhx[0] = z[0] - r;
  zmhx[1] = ModPosNegPi(z[1] - phi);
  zmhx[2] = z[2] - rdot;
  return zmhx;
}

// Measurement residual for laser
VectorXd zminushx_laser(const VectorXd& z, const VectorXd& x) {
  return z - x.segment(0,2);
}

// The fusion EKF engine.  
// This object maintains the state of the target, and can update it given a nwew measurement
FusionEKF::FusionEKF() {
  // Has the state been initialized yet?
  is_initialized_ = false;

  // Keep track of teh time of the most recent measurement update
  previous_timestamp_ = 0;

  // Fixed measurement noise covariance matrix - laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  // Fixed measurement covariance matrix - radar
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  // Fixed linear measurement matrix for laser
  H_laser_ = MatrixXd::Identity(2, 4);

  // Set the process noise uncertainty
  s2ax_ = 9.0;
  s2ay_ = 9.0;

}


// Given a new measurement (either radar or lidar), process the measurement and update the state
void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  // Get the timestamp of the measurememnt
  const long long timestamp = measurement_pack.timestamp_;
  
  // If this is the first measurement, we have to initialize the state
  if (!is_initialized_) {

    x_ = VectorXd::Zero(4);
    P_ = MatrixXd::Identity(4,4);
    P_(2,2) = 100.0;
    P_(3,3) = 100.0;  // we assume that we are more uncertain about velocity

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Initialize using a radar measurement
      const double r = measurement_pack.raw_measurements_[0];
      const double phi = measurement_pack.raw_measurements_[1];
      const double rdot = measurement_pack.raw_measurements_[2];
      x_[0] = r*cos(phi);
      x_[1] = r*sin(phi);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize using a laser measuremement
      x_.segment(0,2) = measurement_pack.raw_measurements_;
    }
    else {
      std::cout << "UNRECOGNIZED measuremement type : " << measurement_pack.sensor_type_ << std::endl;
    }

    // That's all folks.  Record the time and return.
    is_initialized_ = true;
    previous_timestamp_ = timestamp;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  // How much time elapsed since the last update in seconds
  const double dt_s = double(timestamp - previous_timestamp_) / 1.0e6;
  // Make the linear prediction matrix
  MatrixXd F = MatrixXd::Identity(4,4);
  F(0,2) = dt_s;
  F(1,3) = dt_s; 
  // And the process noise
  MatrixXd Q = makeProcessNoiseCovariance(dt_s,s2ax_,s2ay_);

  // Call the predict function to push x_ and P_ forward to the current time
  kalman_filter::Predict(x_,P_,F,Q);

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  // unpack the measureement data
  const VectorXd z = measurement_pack.raw_measurements_;
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // process a radar measurement (have to compute the Jacobian for nonlinear measurement function)
    const MatrixXd H_radar = tools.CalculateJacobian(x_); 
    kalman_filter::Update(x_,P_,H_radar,R_radar_,zminushx_radar(z,x_));
  } 
  else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) { 
    // process a laser measurement (measurement function is linear)
    kalman_filter::Update(x_,P_,H_laser_,R_laser_,zminushx_laser(z,x_));
  }

  // udpate the clock
  previous_timestamp_ = timestamp;

}
