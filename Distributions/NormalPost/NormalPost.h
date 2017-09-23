#ifndef ML_NORMAL_POST_H
#define ML_NORMAL_POST_H

#include "../../Modelo.h"
#include "../Normal/Normal.h"


struct NormalPost : Normal
{
	NormalPost (double alpha_ = 1.0, double beta_ = 1.0, double gamma_ = 1.0, double delta_ = 0.0)
	{
		params(alpha_, beta_, gamma_, delta_);
	}


	void params (double alpha_, double beta_ = 1.0, double gamma_ = 1.0, double delta_ = 0.0)
	{
		alpha = alpha_;
		beta = beta_;
		gamma = gamma_;
		delta = delta_;

		fit(0, 0, 0);

		Normal::update();
	}

	auto params ()
	{
		return make_tuple(mu, sigma, alpha, beta, gamma, delta);
	}


	void fit (const Vec& x)
	{
		double sum = x.sum();

		fit(sum, pow(x.array() - (sum / x.rows()), 2).sum(), x.rows());
	}

	void fit (double sum, double sumSquare, int M)
	{
		mu = (sum + gamma * delta) / (M + gamma);

		sigma = (sumSquare + 2 * beta + gamma * pow(delta - mu, 2)) / (M + 3 + 2 * alpha);

		Normal::update();
	}



	double alpha;
	double beta;
	double gamma;
	double delta;
}; 




#endif