#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_
#include "Eigen/Dense"

namespace kalman_filter {

// Straight forwad Kalman filter predict step
void Predict(Eigen::VectorXd &x, Eigen::MatrixXd &P, const Eigen::MatrixXd &F, const Eigen::MatrixXd &Q);

// Straight forward Kalman filter update step
void Update(Eigen::VectorXd &x, Eigen::MatrixXd &P, 
  const Eigen::MatrixXd &H, const Eigen::MatrixXd &R, 
  const Eigen::VectorXd& yk);


}   // namespace kalman_filter


#endif /* KALMAN_FILTER_H_ */
