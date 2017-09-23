#ifndef ML_BERNOULLI_MIXTURE_DISTRIBUTION_H
#define ML_BERNOULLI_MIXTURE_DISTRIBUTION_H

#include "../../Modelo.h"

#include "../Multinomial/Multinomial.h"


struct BernoulliMixture : public Multinomial
{
	BernoulliMixture (int N_ = 2, int K_ = 2)
	{
		params(N, K);
	}

	BernoulliMixture (const Mat& mu_, const Vec& alpha_)
	{
		params(mu_, alpha_);
	}



	void fit (const Mati& X, int K = 2, double eps = 1e-2, int maxIter = 10)
	{
		int M = X.rows(), N = X.cols();

		Multinomial::mu = Vec::NullaryExpr(K, [&](int){ return 1.0 / K; });

		mu = Mat::NullaryExpr(K, N, [&](int, int){ return rd(0.25, 0.75); });
		
		//mu.rowwise().array() /= mu.rowwise().sum().array();

		for(int i = 0; i < K; ++i)
			mu.row(i) /= mu.row(i).sum();


		update();
		Multinomial::update();


		MatrixXd rik(M, K);


		logMup = log(mu.array()).matrix();
		logMun = log(1.0 - mu.array()).matrix();

		double ll = logLikelihood(X, K), oldLL;


		DB(ll);

		do
		{
			oldLL = ll;


			for(int i = 0; i < M; ++i)
			{
				for(int j = 0; j < K; ++j)
				{
					rik(i, j) = Multinomial::mu(j) * this->operator()(X.row(i), j);

					if(isinf(rik(i, j)) || isnan(rik(i, j))) DB(mu.row(j) << "      Melda"), exit(0);
				}

				//DB(rik.row(i).sum());

				rik.row(i) /= rik.row(i).sum();
			}

			mu.setZero();
			Multinomial::mu.setZero();


			for(int i = 0; i < K; ++i)
			{
				double sum = 0.0;

				for(int j = 0; j < M; ++j)
				{
					mu.row(i) += rik(j, i) * X.row(j).cast<double>();
					sum += rik(j, i);
				}

				mu.row(i) /= sum;

				Multinomial::mu(i) = sum;
			}

			Multinomial::mu /= Multinomial::mu.sum();


			// DB(mu.row(3).head(100));
			// DB(Multinomial::mu(3));


			update();
			Multinomial::update();


			ll = logLikelihood(X, K);

			DB(ll);

			logMup = log(mu.array()).matrix();
			logMun = log(1.0 - mu.array()).matrix();

		//} while(abs(ll - oldLL) > eps && maxIter--);
		} while(maxIter--);

	}	


	double logLikelihood (const Mati& X, int K)
	{
		double r = 0.0;

		for(int i = 0; i < X.rows(); ++i)
			r += log(this->operator()(X.row(i)));

		return r;
	}




	void params (int N_, int K_)
	{
		mu = Mat::Constant(K_, N, 0.0);

		Multinomial::params(K_);

		update();
	}

	void params (const Mat& mu_, const Vec& alpha_)
	{
		mu = mu_;

		Multinomial::params(alpha_);

		update();
	}

	auto params ()
	{
		return make_pair(mu, Multinomial::mu);
	}

	void update ()
	{
		N = mu.cols();
	}



	double operator () (const Veci& x, int k)
	{
		double r = 0.0;

		for(int i = 0; i < N; ++i)
			r += x(i) ? logMup(k, i) : logMun(k, i);
			//r *= (x(i) ? mu(k, i) : 1.0 - mu(k, i));

		r = exp(r / 10.0);


		return r;
	}

	double operator () (const Veci& x)
	{
		double r = 0.0;

		for(int i = 0; i < K; ++i)
			r += Multinomial::mu(i) * this->operator()(x, i);

		return r;
	}

	Vec operator () ()
	{
		int k = Multinomial::operator()();

		Vec x(N);

		for(int i = 0; i < N; ++i)
			x(i) = rd(0.0, 1.0) < mu(k, i);

		return x;
	}



	// double mean ()
	// {
	// }

	// double variance ()
	// {
	// }

	// int mode ()
	// {
	// }



	Mat mu;

	Mat logMup, logMun;

	int N;

	RandDouble rd;

};




#endif // ML_BERNOULLI_MIXTURE_DISTRIBUTION_H