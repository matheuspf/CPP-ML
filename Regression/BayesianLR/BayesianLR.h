#ifndef ML_BAYESIAN_LR_H
#define ML_BAYESIAN_LR_H

#include "../../Modelo.h"


struct BayesianLR
{
	BayesianLR () {}

	BayesianLR& fit (Mat X, const Vec& y)
	{
		assert(X.rows() == y.rows() && "Number of dimensions between 'X' and 'y' differ");

		X.conservativeResize(Eigen::NoChange, X.cols() + 1);
		X.col(X.cols() - 1).array() = 1.0;


		M = X.rows(), N = X.cols();

		Mat In = Mat::Identity(N, N);
		In(N-1, N-1) = 0.0;		


		optimizeEvidence(X, y);


		sigma = (beta * X.transpose() * X + alpha * In).inverse();
		
		mu = beta * sigma * X.transpose() * y;


		return *this;
	}




	double predict (Vec x)
	{
		x.conservativeResize(N);
		x(N - 1) = 1.0;

		return mu.dot(x);
	}

	Vec predict (Mat X)
	{
		X.conservativeResize(Eigen::NoChange, N);
		X.col(N - 1).array() = 1.0;

		return X * mu;
	}



	void optimizeEvidence (const Mat& X, const Vec& y, double eps = 1e-8, int maxIter = 100)
	{
		Mat Xt = X.transpose() * X;

		EigenSolver<Mat> es(Xt);

		Vec eigVal = es.eigenvalues().real();

		Mat In = Mat::Identity(N, N);
		In(N-1, N-1) = 0.0;


		alpha = 1.0, beta = 1.0;

		double oldAlpha, oldBeta;
		
		do
		{
			oldAlpha = alpha, oldBeta = beta;

			Mat A = alpha * In + beta * Xt;

			Vec m = beta * A.inverse() * X.transpose() * y;

			double gamma = ((beta * eigVal.array()) / (alpha + beta * eigVal.array())).sum();


			alpha = gamma / (m.dot(m));

			beta = (N - gamma) / (y - X * m).squaredNorm();

			
		} while((abs(alpha - oldAlpha) + abs(beta - oldBeta)) / (alpha + beta) > 2*eps && maxIter--);
	}





	// double infer (const Vec& x, double y)
	// {
	// 	Vec z = x;
	// 	z.conservativeResize(z.rows()+1);
	// 	z(z.rows()-1) = 1.0;

	// 	return gaussian(y, predict(x), z.transpose() * A * z + sigma);
	// }

	// double infer (const Mat& X, const Vec& y)
	// {
	// 	return gaussian(y, predict(X), (X * A * X.transpose()).diagonal().array() + sigma);
	// }


	// double variance (const Vec& x)
	// {
	// 	Vec z = x;
	// 	z.conservativeResize(z.rows()+1);
	// 	z(z.rows()-1) = 1.0;

	// 	return z.transpose() * A * z + sigma;
	// }

	// double gaussian (double x, double mu, double sig)
	// {
	// 	double diff = x - mu;

	// 	return (1.0 / (sqrt(2*pi()*sig))) * exp(-(0.5 / sig) * diff * diff);
	// }

	// double gaussian (const Vec& x, const Vec& mu, const Mat& sig)
	// {
	// 	Vec diff = x - mu;

	// 	return (1.0 / pow(2*pi()*sig.determinant(), mu.rows() / 2.0)) *
	// 		   exp(-0.5 * diff.transpose() * sig.inverse() * diff);
	// }

	



	int M, N;

	double alpha;

	double beta;
	
	Mat sigma;

	Vec mu;
};



#endif // ML_BAYESIAN_LR_H