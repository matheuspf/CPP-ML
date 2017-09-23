#ifndef ML_BERNOULLI_DISTRIBUTION_H
#define ML_BERNOULLI_DISTRIBUTION_H

#include "../../Modelo.h"


struct Bernoulli
{
	Bernoulli (double mu_ = 0.5) : gen(random_device{}()), rng(0.0, 1.0)
	{
		params(mu_);
	}



	void fit (const Veci& x)
	{
		mu = x.array().count() / double(x.rows());
	}



	void params (double mu_)
	{
		mu = mu_;
	}

	double params ()
	{
		return mu;
	}



	double operator () (int x)
	{
		return x ? mu : 1.0 - mu;
	}

	int operator () ()
	{
		return rng(gen) < mu ? 1 : 0;
	}



	double mean ()
	{
		return mu;
	}

	double variance ()
	{
		return mu * (1.0 - mu);
	}

	int mode ()
	{
		return mu >= 0.5;
		//return mu > 0.5 ? 1 : mu == 0.5 ? rng(gen) >= 0.5 : 0; 
	}



	double mu;

	mt19937 gen;

	uniform_real_distribution<double> rng;

};




#endif // ML_BERNOULLI_DISTRIBUTION_H