#ifndef ML_BINOMIAL_A_POSTERIORI_DISTRIBUTION_H
#define ML_BINOMIAL_A_POSTERIORI_DISTRIBUTION_H

#include "../../Modelo.h"
#include "../BernoulliPost/BernoulliPost.h"
#include "../Binomial/Binomial.h"


struct BinomialPost : public impl::Binomial<BernoulliPost>
{
	using Base = impl::Binomial<BernoulliPost>;

	BinomialPost (int N_ = 1, double mu_ = 0.5, double a_ = 1.0, double b_ = 1.0)
	{
		Base::params(N_);
		BernoulliPost::params(mu_, a_, b_);
	}



	void params (int N_ = 1, double mu_ = 0.5, double a_ = 1.0, double b_ = 1.0)
	{
		Base::params(N_);
		BernoulliPost::params(mu_, a_, b_);
	}

	void params (double mu_, double a_, double b_)
	{
		BernoulliPost::params(mu_, a_, b_);
	}

	void params (double a_, double b_)
	{
		BernoulliPost::params(a_, b_);
	}

	void params (double mu_)
	{
		BinomialPost::params(mu_);
	}


	auto params ()
	{
		return make_tuple(N, mu, a, b);
	}

};


#endif // ML_BINOMIAL_A_POSTERIORI_DISTRIBUTION_H