#ifndef OPT_TRUST_REGION_H
#define OPT_TRUST_REGION_H

#include "../../Modelo.h"
#include "../FiniteDifference.h"


template <class Direction>
struct TrustRegion
{
	template <class Function, class Gradient, class Hessian>
	Vec operator () (Function function, Gradient gradient, Hessian hessian, Vec x)
	{
		double delta = delta0;

		double fx = function(x);
		Vec gx = gradient(x);
		Mat hx = hessian(x);


		for(int iter = 0; iter < maxIter; ++iter)
		{
			Vec dir = static_cast<Direction&>(*this).direction(function, gradient, hessian, x, delta, fx, gx, hx);

			Vec y = x + dir;
			double fy = function(y);

			double rho = (fx - fy) / (-gx.dot(dir) - 0.5 * dir.dot(hx * dir));


			if(rho < EPS && delta == maxDelta)
				return x;

			if(rho < alpha)
				delta = alpha * delta;

			else if(rho > 1.0 - alpha && abs(dir.norm() - delta) < 1e-4)
				delta = min(beta * delta, maxDelta);

			if(delta < EPS)
				return x;

			if(rho > eta)
			{
				x = y;
				fx = fy;
				gx = gradient(x);
				hx = hessian(x);
			}
		}

		return x;
	}



	template <class Function, class Gradient>
	Vec operator () (Function f, Gradient g, const Vec& x)
	{
		return this->operator()(f, g, hessianFD(f), x);
	}

	template <class Function>
	Vec operator () (Function f, const Vec& x)
	{
		return this->operator()(f, gradientFD(f), x);
	}



	double delta0;
	double alpha;
	double beta;
	double eta;

	int maxIter;
	double maxDelta;


private:

	TrustRegion (double delta0 = 1.0, double alpha = 0.25, double beta = 2.0,
			 double eta = 0.1, int maxIter = 1e3, double maxDelta = 1e2) :
			 delta0(delta0), alpha(alpha), beta(beta), eta(eta),
			 maxIter(maxIter), maxDelta(maxDelta) {}


	friend Direction;
};



#endif // OPT_TRUST_REGION_H