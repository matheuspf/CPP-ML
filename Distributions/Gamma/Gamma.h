#ifndef ML_GAMMA_DISTRIBUTION_H
#define ML_GAMMA_DISTRIBUTION_H

#include "../../Modelo.h"
#include "../../LineSearch/Newton.h"

#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>


struct Gamma
{
	Gamma (double alpha_ = 1.0, double beta_ = 1.0) : gen(random_device{}())
	{
		params(alpha_, beta_);
	}


	void params (double alpha_, double beta_ = 1.0)
	{
		alpha = alpha_;
		beta = beta_;

		dist = gamma_distribution<double>(alpha, 1.0 / beta);

		C = exp(alpha * log(beta) - lgamma(alpha));
	}

	auto params ()
	{
		return make_tuple(alpha, beta);
	}

	

	void fit (const Vec& x, bool opt = false)
	{
		int M = x.size();

		// Method of moments: mean and mean^2

		double X = x.mean();
		double Y = pow(x.array(), 2).mean();

		Vec x0(2);

		x0(1) = X / (Y - X * X);
		x0(0) = X * x0(1);


		// For fine tunning

		if(opt)
		{
			double sum = x.sum();
			double logSum = log(x.array()).sum();

			opt::Newton<> solver;

			x0 = solver([&](const Vec& y){ return gammaFunc(y, M, logSum, sum); },
				   		[&](const Vec& y){ return gammaGrad(y, M, logSum, sum); },
				   		[&](const Vec& y){ return gammaHess(y, M); },
				   		x0);
		}


		params(x0(0), x0(1));
	}



	double operator () (double x)
	{
		return C * pow(x, alpha - 1) * exp(-beta * x);
	}

	double operator () ()
	{
		return dist(gen);
	}



	double mean ()
	{
		return alpha / beta;
	}

	double variance ()
	{
		return alpha / (beta * beta);
	}

	double mode ()
	{
		return (alpha - 1) / beta;
	}




	static double gammaFunc (const Vec& x, int M, double logSum, double sum)
	{
		double a = x(0), b = x(1);

		return -(M * (a * log(b) - lgamma(a)) + (a + 1) * logSum - b * sum);
	}

	static Vec gammaGrad (const Vec& x, int M, double logSum, double sum)
	{
		double a = x(0), b = x(1);

		Vec g(2);

		g(0) = -(M * (log(b) - boost::math::digamma(a)) + logSum);
		g(1) = -(M * (a / b) - sum);


		return g;
	}

	static Mat gammaHess (const Vec& x, int M)
	{
		Mat hess = Mat::Constant(2, 2, 0.0);

		hess(0, 0) = M * boost::math::trigamma(x(0));
		hess(0, 1) = hess(1, 0) = -(M / x(1));
		hess(1, 1) = M * (x(0) / (x(1)*x(1)));

		return hess;
	}







	double alpha;
	double beta;

	double C;

	mt19937 gen;

	gamma_distribution<double> dist;

};



#endif // ML_GAMMA_DISTRIBUTION_H