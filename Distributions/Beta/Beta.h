#ifndef ML_BETA_DISTRIBUTION_H
#define ML_BETA_DISTRIBUTION_H

#include "../../Modelo.h"

#include "../../Optimization/Newton/Newton.h"
//#include "../../Optimization/LineSearch/Goldstein/Goldstein.h"

#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>


struct Beta
{

	Beta (double a_ = 1.0, double b_ = 1.0) : gen(random_device{}())
	{
		params(a_, b_);
	}



	void fit (const Vec& data, bool fineTunning = false)
	{
		int N = data.rows();

		double sumA = log(data.array()).sum();
		double sumB = log((1.0 - data.array())).sum();


		double X = data.mean();
		double Y = pow(data.array() - X, 2).mean();

		a = X * (((X * (1 - X)) / Y) - 1);
		b = a * ((1 - X) / X);

		if(fineTunning)
		{
			Vec x0(2);

			x0 << a, b;


			Newton<> solver;

			Vec r = solver([&](const Vec& x){ return betaLL(x, N, sumA, sumB); },
						   [&](const Vec& x){ return betaGrad(x, N, sumA, sumB); },
						   betaHess, x0);

			a = r(0);
			b = r(1);
		}

		update();
	}



	void params (double a_, double b_)
	{
		a = a_;
		b = b_;

		update();
	}

	void update ()
	{
		gammaX = gamma_distribution<double>(a, 1.0);
		gammaY = gamma_distribution<double>(b, 1.0);
	}

	auto params ()
	{
		return make_tuple(a, b);
	}



	double operator () (double x)
	{
		return (1.0 / beta(a, b)) * pow(x, a-1.0) * pow(1.0 - x, b-1.0);
	}

	double operator () ()
	{
		double gx = gammaX(gen);

		return gx / (gx + gammaY(gen));
	}



	double mean ()
	{
		return a / (a + b);
	}

	double variance ()
	{
		return (a * b) / (pow(a + b, 2) * (a + b + 1.0));
	}

	double mode ()
	{
		return (a - 1) / (a + b - 2.0);
	}




	static double betaLL (const Vec& x, int N, double sumA, double sumB)
	{
		return -(N * (lgamma(x(0) + x(1)) - lgamma(x(0)) - lgamma(x(1))) + (x(0) - 1) * sumA + (x(1) - 1) * sumB);
	}

	static Vec betaGrad (const Vec& x, int N, double sumA, double sumB)
	{
		using boost::math::digamma;


		Vec g(2);

		double digab = digamma(x(0) + x(1));

		g(0) = (digamma(x(0)) - digab - (1.0 / N) * sumA);
		g(1) = (digamma(x(1)) - digab - (1.0 / N) * sumB);

		return g;
	}

	static Mat betaHess (const Vec& x)
	{
		using boost::math::trigamma;


		Mat h(2, 2);

		double triab = trigamma(x(0) + x(1));

		h(0, 0) = (trigamma(x(0)) - triab);
		h(0, 1) = h(1, 0) = (-triab);
		h(1, 1) = (trigamma(x(1)) - triab);

		return h;
	}





	double a;
	double b;

	mt19937 gen;

	gamma_distribution<double> gammaX;
	gamma_distribution<double> gammaY;

};




#endif // ML_BETA_DISTRIBUTION_H