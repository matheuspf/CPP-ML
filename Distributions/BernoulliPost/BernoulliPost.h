#ifndef ML_BERNOULLI_POST_DISTRIBUTION_H
#define ML_BERNOULLI_POST_DISTRIBUTION_H

#include "../../Modelo.h"
#include "../Bernoulli/Bernoulli.h"


struct BernoulliPost : public Bernoulli
{
	BernoulliPost (double mu_ = 0.5) : Bernoulli(mu_)
	{
		fit(mu, 1.0);
	}

	BernoulliPost (double mu_, double a_, double b_) : Bernoulli(mu_)
	{
		params(a_, b_);
	}

	BernoulliPost (double a_, double b_)
	{
		params(a_, b_);
	}



	void fit (double count, double total)
	{
		mu = (count + a - 1.0) / (total + a + b - 2.0);
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




#endif // ML_BERNOULLI_POST_DISTRIBUTION_H