#ifndef ML_BERNOULLI_BAYES_DISTRIBUTION_H
#define ML_BERNOULLI_BAYES_DISTRIBUTION_H

#include "../../Modelo.h"
#include "../Bernoulli/Bernoulli.h"


struct BernoulliBayes : public Bernoulli
{
	BernoulliBayes (double mu_ = 0.5) : Bernoulli(mu_)
	{
		fit(mu, 1.0);
	}

	BernoulliBayes (double mu_, double a_, double b_) : Bernoulli(mu_)
	{
		params(a_, b_);
	}

	BernoulliBayes (double a_, double b_)
	{
		params(a_, b_);
	}



	void fit (double count, double total)
	{
		mu = (count + a) / (total + a + b);
	}

	void fit (const Veci& x)
	{
		fit(x.array().count(), x.rows());
	}
	

	void params (double mu_)
	{
		Bernoulli::params(mu_);

		fit(mu, 1.0);
	}

	void params (double mu_, double a_, double b_)
	{
		Bernoulli::params(mu_);

		params(a_, b_);
	}

	void params (double a_, double b_)
	{
		a = a_;
		b = b_;

		fit(mu, 1.0);
	}

	auto params ()
	{
		return make_tuple(mu, a, b);
	}



	double a = 1.0;
	double b = 1.0;
};




#endif // ML_BERNOULLI_BAYES_DISTRIBUTION_H