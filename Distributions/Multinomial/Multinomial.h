#ifndef ML_MULTINOMIAL_DISTRIBUTION_H
#define ML_MULTINOMIAL_DISTRIBUTION_H

#include "../../Modelo.h"


struct Multinomial
{
	Multinomial (int K_)
	{
		params(K_);
	}

	template <class V = Vec, enable_if_t<is_same_v<decay_t<V>, Vec>, int> = 0>
	Multinomial (V&& mu_ = V::Constant(1, 0.0)) : gen(random_device{}()), rng(0.0, 1.0)
	{
		params(forward<V>(mu_));
	}



	void fit (const Veci& x)
	{
		int minCoeff = x.minCoeff(), maxCoeff = x.maxCoeff();

		assert(minCoeff >= 0 && maxCoeff < 1e7);


		K = maxCoeff + 1;

		mu.conservativeResize(K);
		fill(mu.data(), mu.data() + mu.rows(), 0.0);

		cumSum.resize(K);
		cumSum[0] = -1;


		for_each(x.data(), x.data() + x.rows(), [&](int a){
			mu[a]++;
		});
			

		mu /= x.rows();
	}



	
	void params (const std::vector<int>& v)
	{
		Vec mu_(v.size());

		for(int i = 0; i < v.size(); ++i)
			mu_(i) = v[i];

		params(mu_);
	}

	
	void params (Vec mu_)
	{
		mu = mu_;

		update();
	}

	void params (int K_)
	{
		mu.conservativeResize(K_);

		update();
	}


	void update ()
	{
		mu /= mu.sum();

		K = mu.size();

		cumSum.resize(K);
		cumSum[0] = -1;
	}



	Vec params ()
	{
		return mu;
	}



	double operator () (int x)
	{
		return mu[x];
	}

	// double operator () (const Veci& x)
	// {
	// 	return accumulate(x.data(), x.data() + x.rows(), 1.0,
	// 					  [&](double sum, int a){ return sum * operator()(a); });
	// }

	double operator () (const Veci& x)
	{
		return operator()(find_if(x.data(), x.data() + x.rows(), [](int a){ return a; }) - x.data());
	}

	int operator () ()
	{
		if(cumSum[0] == -1)
		{
			cumSum[0] = operator()(0);

			for(int i = 1; i < K; ++i)
				cumSum[i] = cumSum[i-1] + operator()(i);
		}


		return lower_bound(cumSum.begin(), cumSum.end(), rng(gen)) - cumSum.begin();
	}



	Vec mean ()
	{
		return mu;
	}

	// double variance ()
	// {
	// }

	// Veci mode ()
	// {
	// }



	int K;

	Vec mu;

	mt19937 gen;

	uniform_real_distribution<double> rng;

	vector<double> cumSum;

};




#endif // ML_MULTINOMIAL_DISTRIBUTION_H