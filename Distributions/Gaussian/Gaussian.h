#ifndef ML_GAUSSIAN_DISTRIBUTION_H
#define ML_GAUSSIAN_DISTRIBUTION_H

#include "../../Modelo.h"
//#include <Eigen34/unsupported/Eigen/MatrixFunctions>


struct Gaussian
{
	template <class V, class M>
	Gaussian (V&& mu_, M&& sigma_) : gen(random_device{}()), rng(0.0, 1.0)
	{
		params(forward<V>(mu_), forward<M>(sigma_));
	}

	template <class V>
	Gaussian (V&& mu_, double sigma_ = 1.0) : Gaussian(forward<V>(mu_), sigma_ * Mat::Identity(mu_.rows(), mu_.rows())) {}


	Gaussian (int N_ = 1) : Gaussian(Vec::Constant(N_, 0.0), 1.0) {}




	void fit (Mat X)
	{
		mu = X.colwise().mean();

		X = X.rowwise() - mu.transpose();

		sigma = (1.0 / (X.rows() - 1)) * X.transpose() * X;

		update();
	}




	template <class V, class M>
	void params (V&& mu_, M&& sigma_)
	{
		mu = forward<V>(mu_);
		sigma = forward<M>(sigma_);

		update();
	}


	void update ()
	{
		N = mu.rows();


		LLT<Mat> solver(sigma);

		sigma = solver.matrixL();

		sigmaInv = solver.solve(Mat::Identity(N, N));

		determinant = sigma.triangularView<Lower>().determinant();

		C = 1.0 / (pow(2 * pi(), N / 2.0) * determinant);
	}



	auto params ()
	{
		return make_tuple(mu, sigma * sigma.transpose());
	}


	double operator () (const Vec& x)
	{
		return C * exp(-0.5 * (x - mu).transpose() * sigmaInv * (x - mu));
	}


	Mat operator () (int M)
	{
		return (sigma * Mat::NullaryExpr(N, M, [&](int, int){ return rng(gen); })).transpose().rowwise() + mu.transpose();
	}

	Vec operator () ()
	{
		return (sigma * Vec::NullaryExpr(N, [&](int){ return rng(gen); })) + mu;
	}



	Vec mean ()
	{
		return mu;
	}

	Mat variance ()
	{
		return sigma;
	}

	Vec mode ()
	{
		return mean();
	}




	double logLikelihood (const Mat& X)
	{
		double r = 0.0;

		for(int i = 0; i < X.rows(); ++i)
			r += log(operator()(X.row(i)));

		return r;
	}




	Vec mu;
	Mat sigma, sigmaInv;

	int N;

	mt19937 gen;

	normal_distribution<double> rng;


	double C, determinant;
};




#endif // ML_GAUSSIAN_DISTRIBUTION_H