#ifndef ML_NORMAL_BAYESIAN_DISTRIBUTION_H
#define ML_NORMAL_BAYESIAN_DISTRIBUTION_H

#include "../../Modelo.h"
#include "../NormalPost/NormalPost.h"


struct NormalBayes : NormalPost
{
	NormalBayes (double alpha_ = 1.0, double beta_ = 1.0, double gamma_ = 1.0, double delta_ = 0.0) :
				 NormalPost(alpha_, beta_, gamma_, delta_)
	{
		fit();
	}


	void fit (const Vec& x)
	{
		fit(x.rows(), x.sum(), pow(x.array(), 2.0).sum() );
	}


	void fit (int M = 0, double sum = 0.0, double sumSquared = 0.0)
	{
		beta = (sumSquared / 2.0) + beta + ((gamma * pow(delta, 2)) / 2.0) -
			   (pow(gamma * delta + sum, 2) / (2 * (gamma + M)));

		delta = (gamma * sigma + sum) / (gamma + M);

		gamma = gamma + M;

		alpha = alpha + (M / 2.0);


		NormalPost::update();
	}



	double operator () (double x)
	{
		double alphaPred = alpha + 0.5;

		double gammaPred = gamma + 1.0;

		double deltaPred = delta;

		double betaPred = (pow(x, 2) / 2.0) + beta + ((gamma * pow(delta, 2)) / 2.0) - 
						  (pow(gamma * delta + x, 2) / (2 * (gamma + 1.0)));

		return exp(-0.5 * log(2*pi()) +
				   0.5 * log(gamma) + alpha * log(beta) + lgamma(alphaPred) -
				  (0.5 * log(gammaPred) + alphaPred * log(betaPred) + lgamma(alpha)));
	}


	double mode ()
	{
		return ((gamma * delta) / (gamma + 1.0)) / (1.0 + (1.0 / (gamma + 1.0)));
	}

}; 




#endif