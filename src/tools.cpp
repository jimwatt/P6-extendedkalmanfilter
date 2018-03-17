#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

// Compute the RMSE error between estimate and truth
VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

	VectorXd rmse = VectorXd::Zero(4);

	if(estimations.size() != ground_truth.size()
			|| estimations.size() == 0){
		cout << "Invalid estimation or ground_truth data" << endl;
		return rmse;
	}

	//accumulate squared residuals
	for(unsigned int i=0; i < estimations.size(); ++i){
		VectorXd residual = estimations[i] - ground_truth[i];

		//coefficient-wise multiplication
		residual = residual.array()*residual.array();
		rmse += residual;
	}

	//calculate the mean
	rmse = rmse/estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();

	return rmse;

}


// compute the Jacobian fo the Radar measurement function
MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

	MatrixXd Hj = MatrixXd::Zero(3,4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//pre-compute a set of terms to avoid repeated calculation
	float c1 = px*px+py*py;
	float c2 = sqrt(c1);
	float c3 = (c1*c2);

	//check division by zero
	if(fabs(c1) < 0.0001){
		cout << "CalculateJacobian () - Error - Division by Zero" << endl;
		return Hj;
	}

	//compute the Jacobian matrix
	Hj(0,0) =  px/c2;
	Hj(0,1) =  py/c2;
	Hj(1,0) = -py/c1;
	Hj(1,1) =  px/c1;
	Hj(2,0) =  py*(vx*py - vy*px)/c3;
	Hj(2,1) =  px*(px*vy - py*vx)/c3;
	Hj(2,2) =  px/c2;
	Hj(2,3) =  py/c2;

	return Hj;
}
