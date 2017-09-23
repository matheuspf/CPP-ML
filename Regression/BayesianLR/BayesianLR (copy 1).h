#ifndef ML_BAYESIAN_LR_H
#define ML_BAYESIAN_LR_H

#include "../../Modelo.h"
#include "../../LineSearch/SamplingLS.h"
//#include "gnuplot/iostream.h"

using namespace opt;




struct BayesianLR
{
	BayesianLR (double sigmaP = 1e1, double sigBias = 1e-5) : sigmaP(sigmaP), sigBias(sigBias)
	{}



	BayesianLR& learn (Mat X, const Vec& y)
	{
		assert(X.rows() == y.rows() && "Number of dimensions between 'X' and 'y' differ");

		X.conservativeResize(Eigen::NoChange, X.cols()+1);
		X.col(X.cols()-1).array() = 1.0;

		M = X.rows(), N = X.cols();


		Mat In = Mat::Identity(N, N);
		Mat Im = Mat::Identity(M, M);

		//In(N-1, N-1) = N <= M ? sigBias : 1.0 / sigBias;

		sigma = optSigma(X, y);
		
		DB(sigma);


		if(N <= M)
			A = (X.transpose() * X + (sigma / sigmaP) * In).colPivHouseholderQr().solve(Mat::Identity(N, N));

		else
			A = sigma * (sigmaP * In - sigmaP * X.transpose() *
				(X * X.transpose() + (sigma / sigmaP) * Im).colPivHouseholderQr().solve(Mat::Identity(M, M)) * X);


		AXw = A * X.transpose() * y;


		bias = AXw(N-1);

		AXw = Vec(AXw.head(N-1));


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




	// double optSigma (const Mat& X, const Vec& y)
	// {
	// 	auto sigmaFunc = [&](double sig){ return sigFunc(sig, X, y, sigmaP); };
	// 	auto sigmaGrad = [&](double sig){ return sigFunc(sig, X, y, sigmaP); };
	// 	auto sigmaHess = gradientFD(sigmaGrad);

	// 	Newton<BackTrackingLS> ls(BackTrackingLS(1.0));

	// 	return ls(sigmaFunc, sigmaGrad, sigmaHess, 1e-2);
	// }


	double optSigma (const Mat& X, const Vec& y)
	{
		auto func = N <= M ? &BayesianLR::sigFuncN : &BayesianLR::sigFuncM;

		auto sigmaFunc = [&](double sig){ return invoke(func, *this, sig, X, y, sigmaP); };

		SamplingLS ls(lsIter);


		//return ls(sigmaFunc, N <= M ? 1e-4 : sigmaP / 2, pow((y.array() - y.mean()), 2).sum() / y.rows());
		return ls(sigmaFunc, sigmaP / 10, max(2 * sigmaP, pow((y.array() - y.mean()), 2).sum() / y.rows()));
	}



	double sigFuncM (double sig, const Mat& X, const Vec& w, double sigP)
	{
		MatXd cov = (sigP * X * X.transpose() + sig * MatXd::Identity(X.rows(), X.rows()));

		return log(cov.determinant()) + w.transpose() * cov.inverse() * w;
	}


	double sigFuncN (double sig, const Mat& X, const Vec& w, double sigP)
	{
		MatXd cov = (1/sigma) * MatXd::Identity(M, M) - (1/sigma) * X * 
				   (X.transpose() * X + (sigma/sigmaP) * MatXd::Identity(N, N)).colPivHouseholderQr().solve(Mat::Identity(N, N)) * X.transpose();

		return -log(cov.determinant()) + w.transpose() * cov * w;
	}


	double sigGrad (double sig, const Mat& X, const Vec& w, double sigP)
	{
		MatXd cov = (sigP * X * X.transpose() + sig * MatXd::Identity(X.rows(), X.rows())).inverse();

		return cov.trace() - w.transpose() * cov * cov * w;
	}




	int M, N;

	double sigma, sigmaP;
	
	Mat A;

	Vec AXw;

	double bias, sigBias;

	int lsIter = 1e1;
};



#endif // ML_BAYESIAN_LR_H