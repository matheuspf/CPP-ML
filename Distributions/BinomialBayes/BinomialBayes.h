#ifndef ML_BINOMIAL_BAYESIAN_DISTRIBUTION_H
#define ML_BINOMIAL_BAYESIAN_DISTRIBUTION_H

#include "../../Modelo.h"
#include "../BernoulliBayes/BernoulliBayes.h"
#include "../Binomial/Binomial.h"


struct BinomialBayes : public impl::Binomial<BernoulliBayes>
{
	using Base = impl::Binomial<BernoulliBayes>;

	BinomialBayes (int N_ = 1, double mu_ = 0.5, double a_ = 1.0, double b_ = 1.0)
	{
		Base::params(N_);
		BernoulliBayes::params(mu_, a_, b_);
	}



	void params (int N_ = 1, double mu_ = 0.5, double a_ = 1.0, double b_ = 1.0)
	{
		Base::params(N_);
		BernoulliBayes::params(mu_, a_, b_);
	}

	void params (double mu_, double a_, double b_)
	{
		BernoulliBayes::params(mu_, a_, b_);
	}

	void params (double a_, double b_)
	{
		BernoulliBayes::params(a_, b_);
	}

	void params (double mu_)
	{
		BinomialBayes::params(mu_);
	}


	auto params ()
	{
		return make_tuple(N, mu, a, b);
	}

};


#endif // ML_BINOMIAL_BAYESIAN_DISTRIBUTION_H