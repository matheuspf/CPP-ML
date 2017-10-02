#ifndef ML_BAYESIAN_LR_H
#define ML_BAYESIAN_LR_H

#include "../../Modelo.h"


struct BayesianLR
{
	BayesianLR (double sigmaP = 1e1, double sigBias = 1e-8) : sigmaP(sigmaP), sigBias(sigBias) {}



	BayesianLR& fit (Mat X, const Vec& y)
	{
		assert(X.rows() == y.rows() && "Number of dimensions between 'X' and 'y' differ");

		X.conservativeResize(Eigen::NoChange, X.cols()+1);
		X.col(X.cols()-1).array() = 1.0;


		M = X.rows(), N = X.cols();


		Mat In = Mat::Identity(N, N);
		Mat Im = Mat::Identity(M, M);
		In(N-1, N-1) = 0.0;

		optSigmas(X, y);
		//sigmaP = sigma = 1.0;

		db("\n", sigmaP, "   ", sigma, "\n\n");


		//if(N <= M)
			A = (X.transpose() * X + (sigma / sigmaP) * In).colPivHouseholderQr().solve(Mat::Identity(N, N));

		// else
		// 	A = sigma * (sigmaP * In - sigmaP * X.transpose() *
		// 		(X * X.transpose() + (sigma / sigmaP) * Im).colPivHouseholderQr().solve(Mat::Identity(M, M)) * X);

		
		AXw = A * X.transpose() * y;

		bias = AXw(N-1);

		AXw.conservativeResize(N-1);

		//AXw = Vec(AXw.head(N-1));


		return *this;
	}



	double infer (const Vec& x, double y)
	{
		Vec z = x;
		z.conservativeResize(z.rows()+1);
		z(z.rows()-1) = 1.0;

		return gaussian(y, operator()(x), z.transpose() * A * z + sigma);
	}

	double infer (const Mat& X, const Vec& y)
	{
		return gaussian(y, operator()(X), (X * A * X.transpose()).diagonal().array() + sigma);
	}


	double variance (const Vec& x)
	{
		Vec z = x;
		z.conservativeResize(z.rows()+1);
		z(z.rows()-1) = 1.0;

		return z.transpose() * A * z + sigma;
	}



	double operator () (const Vec& x)
	{
		return x.dot(AXw) + bias;
	}

	Vec operator () (const Mat& X)
	{
		return (X * AXw).array() + bias;
	}




	double gaussian (double x, double mu, double sig)
	{
		double diff = x - mu;

		return (1.0 / (sqrt(2*pi()*sig))) * exp(-(0.5 / sig) * diff * diff);
	}

	double gaussian (const Vec& x, const Vec& mu, const Mat& sig)
	{
		Vec diff = x - mu;

		return (1.0 / pow(2*pi()*sig.determinant(), mu.rows() / 2.0)) *
			   exp(-0.5 * diff.transpose() * sig.inverse() * diff);
	}





	void optSigmas (Mat X, const Vec& y, double eps = 1e-8, int maxIter = 10)
	{
		Mat Xt = X.transpose() * X;

		EigenSolver<Mat> es(Xt);

		Vec eigVal = es.eigenvalues().real();

		Mat In = Mat::Identity(N, N);
		In(N-1, N-1) = 0.0;

		double alpha = 1.0, beta = 1.0, oldAlpha, oldBeta;

		do
		{
			oldAlpha = alpha, oldBeta = beta;

			Mat A = alpha * In + beta * Xt;

			Vec m = beta * A.inverse() * X.transpose() * y;

			double gamma = ((beta * eigVal.array()) / (alpha + beta * eigVal.array())).sum();


			alpha = gamma / (m.dot(m));

			beta = (N - gamma) / (y - X * m).dot(y - X * m);

			//db(maxIter, alpha, beta);

		} while((abs(alpha - oldAlpha) + abs(beta - oldBeta)) / (alpha + beta) > 2*eps && maxIter--);

		sigmaP = 1.0 / alpha;
		sigma = 1.0 / beta;
	}



	int M, N;

	double sigma, sigmaP;
	
	Mat A;

	Vec AXw;

	double bias, sigBias;

	int lsIter = 1e1;
};



#endif // ML_BAYESIAN_LR_H