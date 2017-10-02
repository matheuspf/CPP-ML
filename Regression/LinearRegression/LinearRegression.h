#ifndef ML_LINEAR_REGRESSION_H
#define ML_LINEAR_REGRESSION_H

#include "../../Modelo.h"



struct LinearRegression
{
	LinearRegression (double sigmaP = 0.0, double biasFac = 0.0) : sigmaP(sigmaP), biasFac(biasFac) {}



	LinearRegression& fit (Mat X, const Vec& y)
	{
		assert(X.rows() == y.rows() && "Number of dimensions between 'X' and 'y' differ");

		X.conservativeResize(Eigen::NoChange, X.cols()+1);
		X.col(X.cols()-1).array() = 1.0;


		M = X.rows(), N = X.cols() - 1;

		Mat In = Mat::Identity(N+1, N+1);

		In(N, N) = biasFac;


		phi = (X.transpose() * X + sigmaP * In).colPivHouseholderQr().solve(X.transpose() * y);


		
		auto err = y - X * phi;

		sigma = (1.0 / (M - 1)) * err.dot(err.transpose());


		bias = phi(N);

		phi = Vec(phi.head(N));


		return *this;
	}



	double infer (const Vec& x, double y)
	{
		return gaussian(y, predict(x), sigma);
	}

	double infer (const Mat& X, const Vec& y)
	{
		return gaussian(y, predict(X), sigma);
	}



	double predict (const Vec& x)
	{
		return phi.dot(x) + bias;
	}

	Vec predict (const Mat& X)
	{
		return (X * phi).array() + bias;
	}




	double gaussian (double x, double mu, double sigma)
	{
		double diff = x - mu;

		return (1.0 / (sqrt(2*pi()*sigma))) * exp(-(0.5 / sigma) * diff * diff);
	}

	double gaussian (const Vec& x, const Vec& mu, double sigma)
	{
		Vec diff = x - mu;

		return (1.0 / pow(2*pi()*sigma, mu.rows() / 2.0)) * exp(-(0.5 / sigma) * diff.dot(diff));
	}


	int M, N;

	Vec phi;

	double sigma;

	double bias;

	double sigmaP, biasFac;
};



#endif // ML_LINEAR_REGRESSION_H