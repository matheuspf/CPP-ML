#ifndef ML_BINOMIAL_DISTRIBUTION_H
#define ML_BINOMIAL_DISTRIBUTION_H

#include "../../Modelo.h"
#include "../Bernoulli/Bernoulli.h"


namespace impl
{

template <class Base>
struct Binomial : public Base
{
	using Base::mu;
	using Base::rng;
	using Base::gen;
	using Base::Base;


	Binomial (double mu_ = 0.5, int N_ = 1) : Base(mu_)
	{
		params(N_);
	}



	void fit (const Veci& x, int N_)
	{
		fit(x);
		fit(N_);
	}

	void fit (const Veci& x)
	{
		Base::fit(x);
	}

	void fit (int N_)
	{
		params(N_);
	}



	void params (double mu_, int N_ = 1)
	{
		Base::params(mu_);

		params(N_);
	}

	void params (int N_)
	{
		N = N_;

		coef.conservativeResizeLike(Matll::Constant(N+1, N+1, -1));

		cumSum.resize(N+1, -1.0);

		ps.resize(N+1, -1.0);
	}


	auto params ()
	{
		return make_tuple(mu, N);
	}



	double operator () (int x)
	{
		return comb(N, x) * powers(x);
	}


	int operator () ()
	{
		if(cumSum[0] == -1)
		{
			cumSum[0] = operator()(0);

			for(int i = 1; i <= N; ++i)
				cumSum[i] = cumSum[i-1] + operator()(i);
		}


		return lower_bound(cumSum.begin(), cumSum.end(), rng(gen)) - cumSum.begin();
	}


	
	int comb (int m, int x)
	{
		if(x == 0 || m == 0 || x == m)
			return 1;

		return coef(m, x) == -1 ? (coef(m, x) = comb(m-1, x) + comb(m-1, x-1)) : coef(m, x);
	}


	double powers (int x)
	{
		return ps[x] == -1.0 ? (ps[x] = pow(mu, x) * pow(1.0 - mu, N - x)) : ps[x];
	}



	double mean ()
	{
		return N * mu;
	}

	double variance ()
	{
		return N * mu * (1.0 - mu);
	}

	int mode ()
	{
		return operator()(floor((N+1)*mu)) > operator()(ceil((N+1)*mu)-1) ? floor((N+1)*mu) : ceil((N+1)*mu)-1;
	}



	int N;

	Matll coef;

	vector<double> cumSum;

	vector<double> ps;
};


} // namespace impl


using Binomial = impl::Binomial<Bernoulli>;


#endif // ML_BINOMIAL_DISTRIBUTION_H