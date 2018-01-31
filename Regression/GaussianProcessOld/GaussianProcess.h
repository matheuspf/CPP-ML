#ifndef ML_GAUSSIAN_PROCESS_H
#define ML_GAUSSIAN_PROCESS_H

#include "../../Modelo.h"

//#include "../../Optimization/Newton/Newton.h"

//#include "../../Optimization/CG/CG.h"

#include "../../Optimization/MDE/MDE.h"

#include "../../Kernels.h"




template <class Kernel = RBFKernel>
struct GaussianProcess
{
	GaussianProcess (const Kernel& kernel = Kernel()) : kernel(kernel) {}


	void fit (Mat X_, Vec y_)
	{
		X = move(X_);
		y = move(y_);

		X.conservativeResize(Eigen::NoChange, X.cols()+1);
		X.col(X.cols()-1).array() = 1.0;

		M = X.rows(), N = X.cols();

		optSigmas();

		Mat K = kernel(X);

		A = (K + (sigma / sigmaPrior) * Mat::Identity(M, M)).inverse();
		
		AXw = A * K * y;
	}


	double operator () (Vec x)
	{
		x.conservativeResize(x.rows() + 1);
		x(x.rows() - 1) = 1.0;

		Vec k = kernel(X, x);

		return (sigmaPrior / sigma) * (y.dot(k) - k.dot(AXw));
	}

	// Vec operator () (const Mat& X)
	// {
	// }




	// void optSigmas ()
	// {
	// 	// sigma = 1.0;
	// 	// sigmaPrior = 1e5;
	// 	// kernel.gamma = 1.0;
	// 	// return;



	// 	auto func = [&](const Vec& x) -> double
	// 	{
	// 		double sig = x(0), sigP = x(1), gamma = x(2);

	// 		Kernel kernel(gamma);

	// 		Mat A = sigP * kernel(X) + sig * Mat::Identity(X.rows(), X.rows());


	// 		return log(A.determinant()) + y.transpose() * A.inverse() * y;
	// 	};


	// 	CG<> solver;
	// 	//Newton<Goldstein, CholeskyIdentity> solver;

	// 	Vec x = Vec::Constant(3, 1.0);

	// 	x = solver(func, x);

		// sigma = x(0);
		// sigmaPrior = x(1);
		// kernel.gamma = x(2);


	// 	DB(x.transpose());
	// }



	void optSigmas ()
	{
		struct OptFunction : mde::Function<3>
		{
			OptFunction (GaussianProcess<RBFKernel>& gp) : gp(gp)
			{
				lowerBounds = {1e-2, 1e-2, 1e-2};
				upperBounds = {1e0, 1e0, 1e0};
			}
		
			double operator () (const Vector& x)
			{
				double sig = x[0], sigP = x[1], gamma = x[2];
		
				RBFKernel kernel(gamma);
		
				Mat A = sigP * kernel(gp.X) + sig * Mat::Identity(gp.X.rows(), gp.X.rows());
		
		
				return log(A.determinant()) + gp.y.transpose() * A.inverse() * gp.y;
			}
		
			GaussianProcess<RBFKernel>& gp;
		};

		mde::Parameters params;
		params.popSize = 100;
		params.children = 3;
		params.maxIter = 1e1;
		params.debug = true;

		mde::MDE<OptFunction> mde(params, OptFunction(*this));

		auto x = mde();

		sigma = x[0];
		sigmaPrior = x[1];
		kernel.gamma = x[2];

		DB(sigma << "    " << sigmaPrior << "    " << kernel.gamma);
		// DB(mde.function(x));
	}



	
	


	int M;
	int N;


	Mat X;
	Vec y;

	Mat A;

	Vec AXw;

	double sigma;
	double sigmaPrior;

	double bias, biasPrior = 1e-6;

	Kernel kernel;
};





#endif // ML_GAUSSIAN_PROCESS_H