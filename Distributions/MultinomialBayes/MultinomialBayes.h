#ifndef ML_MULTINOMIAL_BAYESIAN_DISTRIBUTION_H
#define ML_MULTINOMIAL_BAYESIAN_DISTRIBUTION_H

#include "../../Modelo.h"
#include "../Multinomial/Multinomial.h"


struct MultinomialBayes : public Multinomial
{
	MultinomialBayes (int K) : MultinomialBayes(Vec::Constant(K, 1.0))
	{
	}

	template <class V = Vec>
	MultinomialBayes (V&& alpha_ = V::Constant(1, 1.0))
	{
		params(forward<V>(alpha_));
	}



	void fit (const Veci& x)
	{
		int minCoeff = x.minCoeff(), maxCoeff = x.maxCoeff();

		assert(minCoeff >= 0 && maxCoeff < 1e7);

		fit(x, maxCoeff + 1);
	}


	void fit (const Veci& x, int K_)
	{
		K = K_;

		mu.conservativeResize(K);
		fill(mu.data(), mu.data() + mu.rows(), 0.0);

		cumSum.resize(K);
		cumSum[0] = -1;


		for_each(x.data(), x.data() + x.rows(), [&](int a){
			mu[a]++;
		});
			

		mu = (mu.array() + alpha.array()) / (alpha0 + x.rows());
	}



	template <class V = Vec>
	void params (V&& alpha_)
	{	
		alpha = forward<V>(alpha_);

		alpha0 = alpha.sum();

		Multinomial::params((alpha.array() - 1.0) / (alpha0 - alpha.rows()));
	}

	auto params ()
	{
		return make_tuple(mu, alpha);
	}


	Vec alpha;

	double alpha0;
};


#endif // ML_MULTINOMIAL_BAYESIAN_DISTRIBUTION_H