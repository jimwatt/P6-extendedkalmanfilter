#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace kalman_filter {

// Update x and P, given the motion model and uncertainty
void Predict(VectorXd &x, MatrixXd &P, const MatrixXd &F, const MatrixXd &Q) {

  x = F * x;
  P = F * P * (F.transpose()) + Q;

}

// Update x and P given the measurement residual
void Update(VectorXd &x, MatrixXd &P, const MatrixXd &H, const MatrixXd &R, const VectorXd& yk) {

  const int nx = x.size();
  const MatrixXd Idx = MatrixXd::Identity(nx,nx);
  const MatrixXd Sk = H * P * (H.transpose()) + R;
  const MatrixXd Kk = P * (H.transpose()) * (Sk.inverse());
  x = x + Kk * yk;
  P = (Idx - Kk*H) * P * (Idx - Kk * H).transpose()  +  Kk * R * Kk.transpose(); // this form helps to maintain symmetric P

}

}   // namespace kalman_filter


