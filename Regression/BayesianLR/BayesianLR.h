#ifndef ML_BAYESIAN_LR_H
#define ML_BAYESIAN_LR_H

#include "../../Modelo.h"

#include "../../Optimization/Newton/Newton.h"

#include "../../Optimization/MDE/MDE.h"


struct BayesianLR
{
	BayesianLR (double sigmaP = 1e1, double sigBias = 1e-8) : sigmaP(sigmaP), sigBias(sigBias)
	{}



	BayesianLR& learn (Mat X, const Vec& y)
	{
		assert(X.rows() == y.rows() && "Number of dimensions between 'X' and 'y' differ");

		X.conservativeResize(Eigen::NoChange, X.cols()+1);
		X.col(X.cols()-1).array() = 1.0;


		M = X.rows(), N = X.cols();


		Mat In = Mat::Identity(N, N);
		Mat Im = Mat::Identity(M, M);
		In(N-1, N-1) = 0.0;

		tie(sigmaP, sigma) = optSigmas(X, y);
		//sigmaP = sigma = 1.0;

		DB(sigmaP << "     " << sigma << "\n\n");


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





	pair<double, double> optSigmas (Mat X, const Vec& y, double eps = 1e-8, int maxIter = 100)
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

			beta = (N - gamma) * (y - X * m).dot(y - X * m);

			//DB(alpha << "    " << beta);


		} while(abs(alpha - oldAlpha) + abs(beta - oldBeta) > 2*eps && maxIter--);


		return make_pair(1.0 / alpha, 1.0 / beta);
	}






	// pair<double, double> optSigmas (const Mat& X, const Vec& y, double eps = 1e-3)
	// {

	// 	Mat Xt = X * X.transpose();

	// 	EigenSolver<Mat> es(Xt);

	// 	ArrayXd eigVal = es.eigenvalues().real().array();

	// 	double alpha = 1.0, beta = 1.0, oldAlpha, oldBeta;

	// 	do
	// 	{
	// 		oldAlpha = alpha, oldBeta = beta;


	// 		Mat A = alpha * Xt + beta * Mat::Identity(M, M);

	// 		Mat A0 = Xt + (beta / alpha) * Mat::Identity(M, M);

	// 		Mat A1 = (alpha / beta) * Xt + Mat::Identity(M, M);


	// 		double g0 = (eigVal / (alpha * eigVal + beta)).sum();

	// 		double g1 = (1.0 / (alpha * eigVal + beta)).sum();

			
	// 		alpha = sqrt(double(y.transpose() * A0.inverse() * y) / g0);

	// 		beta = (double(y.transpose() * A1.inverse() * y) / g1);


	// 	} while(abs(alpha - oldAlpha) > eps && abs(beta - oldBeta) > eps);


	// 	return make_pair(alpha, beta);
	// }










	// pair<double, double> optSigmas (const Mat& X, const Vec& y, double eps = 1e-3)
	// {
	// 	auto func = [&](const Vec& x) -> double
	// 	{
	// 		int M = X.rows(), N = X.cols();
	// 		double alpha = x(0), beta = x(1);

	// 		Mat In = Mat::Identity(N, N);
	// 		In(N-1, N-1) = 0.0;

	// 		Mat A = alpha * In + beta * X.transpose() * X;

	// 		Vec m = beta * A.inverse() * X.transpose() * y;

			
	// 		return (N / 2.0) * log(alpha) + (M / 2.0) * log(beta) - 
	// 			   (beta / 2.0) * (y - X * m).dot(y - X * m) + (alpha / 2.0) * m.dot(m) -
	// 			   0.5 * log(A.determinant()) - (M / 2.0) * log(2.0 * pi());
	// 	};

	// 	auto grad = [&](const Vec& x) -> Vec
	// 	{
	// 		int M = X.rows(), N = X.cols();
	// 		double alpha = x(0), beta = x(1);

	// 		Mat A = alpha * Mat::Identity(N, N) + beta * X.transpose() * X;

	// 		Vec m = beta * A.inverse() * X.transpose() * y;


	// 		EigenSolver<Mat> es(X.transpose() * X);

	// 		ArrayXd eigVal = es.eigenvalues().real().array();



	// 		Vec grad(2);

	// 		grad(0) = (N / (2.0 * alpha)) - 0.5 * m.dot(m) - 0.5 * (1.0 / (eigVal + alpha)).sum();

	// 		grad(1) = (M / (2.0 * beta)) - 0.5 * (y - X * m).dot(y - X * m) - (eigVal / (alpha + eigVal)).sum() / (2.0 * beta);

	// 		return grad;
	// 	};

	// 	auto hess = [&](const Vec& x) -> Mat
	// 	{
	// 		int M = X.rows(), N = X.cols();
	// 		double alpha = x(0), beta = x(1);


	// 		EigenSolver<Mat> es(X.transpose() * X);

	// 		ArrayXd eigVal = es.eigenvalues().real().array();


	// 		Mat hess(2, 2);

	// 		hess(0, 0) = 0.5 * (1.0 / (pow(eigVal, 2) + alpha).sum()) - M / (2.0 * alpha * alpha);
	// 		hess(0, 1) = hess(1, 0) = 0.0;
	// 		hess(1, 1) = (eigVal / (alpha + eigVal)).sum() / (2.0 * beta * beta) - M / (2.0 * beta * beta);

	// 		return hess;
	// 	};



	// 	// Vec x(2);
	// 	// x << 1.0, 1.0;

	// 	// Newton<StrongWolfe, CholeskyIdentity> newton;

	// 	// x = newton(func, x);



	// 	struct OptFunction : mde::Function<3>
	// 	{
	// 		OptFunction (const Mat& X, const Vec& y) : X(X), y(y)
	// 		{
	// 			lowerBounds = {1e-5, 1e-5};
	// 			upperBounds = {1e5, 1e5};
	// 		}
		
	// 		double operator () (const Vector& x)
	// 		{
	// 			static Vec v(2);

	// 			v(0) = x[0], v(1) = x[1];

	// 			return lol(v);
	// 		}


	// 		double lol (const Vec& x)
	// 		{
	// 			int M = X.rows(), N = X.cols();
	// 			double alpha = x(0), beta = x(1);
	
	// 			Mat In = Mat::Identity(N, N);
	// 			In(N-1, N-1) = 0.0;
	
	// 			Mat A = alpha * In + beta * X.transpose() * X;
	
	// 			Vec m = beta * A.inverse() * X.transpose() * y;
	
				
	// 			return (N / 2.0) * log(alpha) + (M / 2.0) * log(beta) - 
	// 				   (beta / 2.0) * (y - X * m).dot(y - X * m) + (alpha / 2.0) * m.dot(m) -
	// 				   0.5 * log(A.determinant()) - (M / 2.0) * log(2.0 * pi());
	// 		}

	// 		const Mat& X;
	// 		const Vec& y;
	// 	};

	// 	mde::Parameters params;
	// 	params.popSize = 100;
	// 	params.children = 5;
	// 	params.maxIter = 500;
	// 	params.debug = true;

	// 	mde::MDE<OptFunction> mde(params, OptFunction(X, y));

	// 	auto x = mde();


	// 	return make_pair(x[0], x[1]);
	// }




	int M, N;

	double sigma, sigmaP;
	
	Mat A;

	Vec AXw;

	double bias, sigBias;

	int lsIter = 1e1;
};



#endif // ML_BAYESIAN_LR_H