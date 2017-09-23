#ifndef ML_NORMAL_DISTRIBUTION_H
#define ML_NORMAL_DISTRIBUTION_H

#include "../../Modelo.h"


struct Normal
{
	Normal (double mu_ = 0.0, double sigma_ = 1.0) : gen(random_device{}())
	{
		params(mu_, sigma_);
	}

	
	void fit (const Vec& x)
	{
		double mu_ = x.mean();

		double sigma_ = (1.0 / (x.rows() - 1)) * pow(x.array() - mu, 2).sum();

		params(mu_, sigma_);
	}


	void params (double mu_, double sigma_)
	{
		mu = mu_;
		sigma = sigma_;

		update();
	}

	void update ()
	{
		dist = normal_distribution<double>(mu, sqrt(sigma));

		C = 1.0 / (sqrt(2*pi()*sigma));
	}


	auto params ()
	{
		return make_tuple(mu, sigma);
	}



	double operator () (double x)
	{
		return C * exp(-pow(x - mu, 2) / (2 * sigma));
	}

	double operator () ()
	{
		return dist(gen);
	}



	double mean ()
	{
		return mu;
	}

	double variance ()
	{
		return sigma;
	}

	double mode ()
	{
		return mu;
	}




	double C;

	double mu;
	double sigma;


	normal_distribution<double> dist;

	mt19937 gen;
};




#endif // ML_NORMAL_DISTRIBUTION_H