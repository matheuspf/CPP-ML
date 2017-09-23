#ifndef ML_DIRICHLET_DISTRIBUTION_H
#define ML_DIRICHLET_DISTRIBUTION_H

#include "../../Modelo.h"
#include "../../LineSearch/Newton.h"

#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>


struct Dirichlet
{
	template <class V>
	Dirichlet (V&& alpha_) : gen(random_device{}())
	{
		params(forward<V>(alpha_));
	}



	void fit (const Mat& X)
	{
		int M = X.rows();

		Vec muSum = log(X.array()).colwise().sum();



		Vec x0 = X.colwise().sum() / X.rows();
		//Vec x0 = Vec::Constant(X.cols(), 2.0);

		
		DB(x0);



		opt::Newton<> solver;

		Vec r = solver([&](const Vec& x){ return dirLL(x, M, muSum); },
					   [&](const Vec& x){ return dirGrad(x, M, muSum); },
					   [&](const Vec& x){ return dirHess(x, M); },
					   x0);


		params(r);
	}



	template <class V>
	void params (V&& alpha_)
	{
		alpha = forward<V>(alpha_);

		K = alpha.size();

		alpha0 = alpha.sum();


		gammaDist.resize(K);

		for(int i = 0; i < K; ++i)
			gammaDist[i] = gamma_distribution<double>(alpha(i), 1.0);



		// C = tgamma((long double)alpha0) / accumulate(alpha.data(), alpha.data() + K, (long double){1.0},
		// 		   [](long double sum, long double a){ return sum * tgamma(a); });

		C = exp(lgamma((long double)alpha0) - accumulate(alpha.data(), alpha.data() + K, (long double){0.0},
				[](long double sum, long double a){ return sum + lgamma(a); }));
	}

	Vec params ()
	{
		return alpha;
	}



	double operator () (const Vec& x)
	{
		return inner_product(x.data(), x.data() + K, alpha.data(), 1.0,
								 [](double sum, double x){ return sum * x; },
								 [](double x, double a){ return pow(x, a - 1.0); });
	}

	Vec operator () ()
	{
		// Vec x(K);

		// transform(gammaDist.begin(), gammaDist.end(), x.data(), [&](auto& dist){ return dist(gen); });

		// return x / x.sum();

		Vec x = Vec::NullaryExpr(K, [&](int i){ return gammaDist[i](gen); });

		return x / x.sum();
	}



	Vec mean ()
	{
		return alpha / alpha0;
	}

	Vec variance ()
	{
		return (alpha.array() * (alpha0 - alpha.array())) / (alpha0 * alpha0 * (alpha0 + 1.0));
	}

	Vec mode ()
	{
		return (alpha.array() - 1.0) / (alpha0 - K);
	}




	static double dirLL (const Vec& x, int M, const Vec& logMu)
	{
		double gammaC = M * (lgamma(x.sum()) - accumulate(x.data(), x.data() + x.rows(), 0.0,
				 						  	   [](double sum, double a){ return sum + lgamma(a); }));

		return -(gammaC + (x.array() - 1.0).matrix().transpose() * logMu);
	}

	static Vec dirGrad (const Vec& x, int M, const Vec& logMu)
	{
		using boost::math::digamma;


		Vec g(x.rows());

		double digSum = digamma(x.sum());

		
		for(int i = 0; i < g.rows(); ++i)
			g(i) = -(M *(digSum - digamma(x(i))) + logMu(i));

		return g;
	}

	static Mat dirHess (const Vec& x, int M)
	{
		using boost::math::trigamma;


		Mat h = Mat::Constant(x.rows(), x.rows(), -M * trigamma(x.sum()));


		for(int i = 0; i < x.rows(); ++i)
			h(i, i) += M * trigamma(x(i));


		return h;
	}





	Vec alpha;

	double alpha0;

	int K;

	long double C;


	mt19937 gen;

	std::vector<gamma_distribution<double>> gammaDist;

};




#endif // ML_DIRICHLET_DISTRIBUTION_H