#ifndef ML_BETA_MIXTURE_DISTRIBUTION_H
#define ML_BETA_MIXTURE_DISTRIBUTION_H

#include "../../Modelo.h"

#include "../Beta/Beta.h"
#include "../Multinomial/Multinomial.h"

// #include "../../Optimization/Newton/Newton.h"

// #include "../../Optimization/LineSearch/Goldstein/Goldstein.h"

#include "../../Optimization/MDE/MDE.h"

#include "../../ZipIter/ZipIter.h"


struct mdeFunc : mde::Function<>
{
	mdeFunc(const Vec& X, int K) : X(X), v(K), K(K), mde::Function<>(2*K, 1.0, 10.0)
	{
	}

	double operator () (const vector<double>& x)
	{
		double r = 0.0;

		for(int k = 0; k < K; ++k)
			v[k].params(x[2*k], x[2*k+1]);

		for(int i = 0; i < X.rows(); ++i)
		{
			double aux = 0.0;

			for(int k = 0; k < K; ++k)
				aux += 0.5 * v[k](X[i]);

			r += log(aux);
		}


		return -r;
	}


	vector<Beta> v;

	const Vec& X;

	int K;
};



struct BetaMixture
{
	BetaMixture (double a = 1.0, double b = 1.0, int K = 2)
	{
		params(a, b, K);
	}


	BetaMixture (const Vec& a_, const Vec& b_, const Vec& alphas_)
	{
		params(a_, b_, alphas_);
	}


	void params (double a, double b, int K)
	{
		betas.resize(K);

		for(auto& beta : betas)
			beta.params(a, b);

		mult.params(K);
	}

	void params (const Vec& as, const Vec& bs, const Vec& alphas_)
	{
		for_each(ZIP_ALL(as, bs, betas), it::unZip([](auto&& a, auto&& b, auto&& beta){
			beta.params(a, b);
		}));

		mult.params(alphas_);
	}

	auto params ()
	{
		return make_tuple(betas, mult);
	}




	void fit (const Vec& X, int K = 2, double eps = 1e-2, int maxIter = 10)
	{
		params(1.0, 1.0, K);

		mult.mu = Vec::Constant(K, 1.0 / K);
		mult.update();





		mde::Parameters params;

		params.maxIter = 1e2;
		params.debug = 1;


		mdeFunc func(X, K);


		mde::MDE<mdeFunc> mde(params, func);


		auto x = mde();


		for(int i = 0; i < K; ++i)
		{
			betas[i].a = x[2*i];
			betas[i].b = x[2*i+1];

			betas[i].update();
		}
	}








/*	void fit (Vec X, int K = 2, double eps = 1e-2, int maxIter = 10)
	{
		int M = X.rows();

		ArrayXd logXp = log(X.array());
		ArrayXd logXn = log(1.0 - X.array());


		params(1.0, 1.0, K);


		mt19937 gen{random_device{}()};

		shuffle(X.data(), X.data() + M, gen);


		for(int k = 0; k < K; ++k)
		{
			// auto seg = X.segment(k * (M / K), M / K);

			// double mean = seg.mean();
			// double var = pow(1.0 - seg.array(), 2).mean();

			// betas[k].a = mean * (((mean * (1 - mean)) / var) - 1);
			// betas[k].b = betas[k].a * ((1 - mean) / mean);

			if(k == 0)
			{
				betas[k].a = 2.0;
				betas[k].b = 2.0;
			}

			else
			{
				betas[k].a = 1.0;
				betas[k].b = 1.0;
			}

			betas[k].update();

			mult.mu[k] = 1.0 / K;
		}

		mult.update();



		Mat rik(M, K);

		double ll = -1e20, oldLL;


		do
		{
			oldLL = ll;

			for(int i = 0; i < M; ++i)
			{
				for(int k = 0; k < K; ++k)
					rik(i, k) = mult(k) * betas[k](X(i));

				rik.row(i) /= rik.row(i).sum();

				for(int k = 0; k < K; ++k)
				{
					//DB(rik(i, k));
					if(isnan(rik(i, k)) || isinf(rik(i, k))) DB(i << "   " << k << "       " << "MELDA"), exit(0);
				}
			}


			for(int k = 0; k < K; ++k)
				mult.mu[k] = rik.col(k).sum();

			mult.update();

			//DB(mult.mu.transpose()); exit(0);


			for(int k = 0; k < K; ++k)
			{
				Vec rk = rik.col(k);
				//Vec rk = Vec::Constant(M, 1.0 / K);

				double rkSum = rk.sum();

				Vec x0(2);

				x0 << betas[k].a, betas[k].b;


				Newton<Goldstein> solver;

				Vec r = solver([&](const Vec& x){ return betaLL(x, rk, logXp, logXn); },
							   [&](const Vec& x){ return betaGrad(x, rk, logXp, logXn); },
							   [&](const Vec& x){ return betaHess(x, rkSum); },
							   x0);

				betas[k].a = r(0);
				betas[k].b = r(1);

				DB("\n" << r.transpose());

				betas[k].update();
			}





			ll = logLikelihood(X);

			DB(ll);

		} while(--maxIter);
		//} while(abs(ll - oldLL) > eps && --maxIter);
	}
*/




	double operator () (double x)
	{
		return accumulate(ZIP_ALL(betas, mult.mu), 0.0,
						  it::unZip([&](double sum, auto& b, double alpha)
						  {
						  		//DB(x << "      " << b.a << "       " << b.b << "        " << b(x));
						  		return sum + alpha * b(x);
						  }));
	}

	double operator () ()
	{
		return betas[mult()]();
	}



	// double mean ()
	// {
	// 	return a / (a + b);
	// }

	// double variance ()
	// {
	// 	return (a * b) / (pow(a + b, 2) * (a + b + 1.0));
	// }

	// double mode ()
	// {
	// 	return (a - 1) / (a + b - 2.0);
	// }



	double logLikelihood (const Vec& X)
	{
		return accumulate(X.data(), X.data() + X.rows(), 0.0,
						  [&](double sum, double x){ return sum + log(this->operator()(x)); });
	}



	static double betaLL (const Vec& x, const Vec& rk, const ArrayXd& logXp, const ArrayXd& logXn)
	{
		return rk.dot((lgamma(x(0) + x(1)) - lgamma(x(0)) - lgamma(x(1)) +
					  (x(0) - 1.0) * logXp + (x(1) - 1.0) * logXn).matrix());
	}

	static Vec betaGrad (const Vec& x, const Vec& rk, const ArrayXd& logXp, const ArrayXd& logXn)
	{
		using boost::math::digamma;


		double digab = digamma(x(0) + x(1));

		Vec g(2);

		g(0) = rk.dot((digab - digamma(x(0)) + logXp).matrix());
		g(1) = rk.dot((digab - digamma(x(1)) + logXn).matrix());

		return -g;
	}

	static Mat betaHess (const Vec& x, double rkSum)
	{
		using boost::math::trigamma;


		Mat h(2, 2);

		double triab = trigamma(x(0) + x(1));

		h(0, 0) = triab - trigamma(x(0));
		h(0, 1) = h(1, 0) = triab;
		h(1, 1) = triab - trigamma(x(1));

		return -rkSum * h;
	}







	Multinomial mult;

	vector<Beta> betas;

};




#endif // ML_BETA_MIXTURE_DISTRIBUTION_H