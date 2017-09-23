#ifndef ML_STUDENT_DISTRIBUTION_H
#define ML_STUDENT_DISTRIBUTION_H

#include "../../Modelo.h"

#include "../Gaussian/Gaussian.h"

#include "../../Optimization/LineSearch/Bracketing/Bracketing.h"

#include "../../Optimization/LineSearch/Brents/Brents.h"

#include <boost/math/special_functions/digamma.hpp>



struct Student : public Gaussian
{
	using Gaussian::Gaussian;
	using Gaussian::params;


	Student (double v_)
	{
		params(v_);
	}

	Student (const Vec& mu_, const Mat& sigma_, double v_) : Gaussian(mu_, sigma_)
	{
		params(v_);
	}




	void params (double v_)
	{
		v = v_;

		update();
	}

	void update ()
	{
		gammaGen = gamma_distribution<>(v / 2.0, 1.0 / (v / 2.0));

		C = exp(lgamma((v + N) / 2.0) - ((N / 2.0) * log(v * pi()) + log(determinant) + lgamma(v / 2.0)));
	}

	auto params ()
	{
		return make_tuple(mu, sigma, v);
	}



	double operator () (const Vec& x)
	{
		auto mult = sigmaInv * (x - mu);

		//return C * exp(-((v + N) / 2) * (log(v + mult.dot(mult)) - log(v)));

		return C * pow(1.0 + mult.dot(mult) / v, -(v + N) / 2.0);
	}

	Vec operator () ()
	{
		double h = gammaGen(gen);

		return (sigma / sqrt(h)) * Vec::NullaryExpr(N, [&](int){ return rng(gen); }) + mu;
	}



	void fit (const Mat& X, double precision = 1e-2, int maxIter = 1e2)
	{
		Bracketing brack;

		Brents ls(1e-4, 1e2);


		int M = X.rows();


		Gaussian::fit(X);

		v = 1e3;

		update();



		Vec eh(M), ehLog(M);

		double ll = -1e13, oldLL;


		do
		{
			oldLL = ll;


			Vec mult(N);

			for(int i = 0; i < M; ++i)
			{
				mult = sigmaInv * (X.row(i) - mu.transpose()).transpose();

				double dot = mult.dot(mult);

				eh(i) = ((v + N) / (v + dot));

				ehLog(i) = boost::math::digamma((v + N) / 2) - log(0.5 * (v + dot));
			}


			mu.setZero();
			sigma.setZero();

			double sum = 0.0;

			for(int i = 0; i < M; ++i)
			{
				sum += eh(i);
				mu += eh(i) * X.row(i);
			}

			mu /= sum;

			for(int i = 0; i < M; ++i)
				sigma += eh(i) * (X.row(i).transpose() - mu) * (X.row(i) - mu.transpose());

			sigma /= sum;

			Gaussian::update();


			//auto func = vFunc(X, eh, ehLog);

			auto func = [&](double a) -> double
			{
				double r = X.rows() * ((a / 2) * log(a / 2) - lgamma(a / 2) - 0.25 * log(determinant) - (N / 2.0) * log(2*pi()));

				Vec mult(N);

				for(int i = 0; i < X.rows(); ++i)
				{
					mult = sigmaInv * (X.row(i).transpose() - mu);

					r += 0.5 * (N * ehLog(i) - mult.dot(mult) * eh(i)) + (a / 2 - 1) * ehLog(i) - (a / 2) * eh(i);
				}

				return -r;
			};


			// double fVal = 1e20;

			// for(double a = 1.0; a < 20.0; a += 0.1)
			// {
			// 	double aux = func(a);

			// 	if(aux < fVal)
			// 		fVal = aux, v = a;
			// }


			//v = ls(func, 1.0, 100.0);
			v = ls(func, brack(func, 1.0, 50.0));

			update();


			ll = logLikelihood(X);

			//DB(v << "    " << ll);

			//display(maxIter, ll);


		} while(abs(ll - oldLL) > precision && --maxIter);



	}



	double logLikelihood (const Mat& X)
	{
		double r = 0.0;

		for(int i = 0; i < X.rows(); ++i)
			r += log(operator()(X.row(i)));

		return r;
	}



	auto vFunc (const Mat& X, const Vec& eh, const Vec& ehLog)
	{
		[&](double a) -> double
		{
			double r = X.rows() * ((a / 2) * log(a / 2) - lgamma(a / 2) - log(determinant) - (N / 2.0) * log(2*pi()));

			Vec mult(N);

			for(int i = 0; i < X.rows(); ++i)
			{
				mult = sigmaInv * (X.row(i) - mu);

				r += 0.5 * (N * ehLog(i) - mult.dot(mult) * eh(i)) + (a / 2 - 1) * ehLog(i) - (a / 2) * eh(i);
			}

			return r;
		};
	}



	void display (int iter, double ll)
	{
		DB("Iteration:  " << iter);
		DB("Likelihood:  " << ll << "\n");

		DB(v << "\n");
		DB(mu.transpose() << "\n");
		DB(sigma.transpose() << "\n\n\n");
	}




	double v = 1e5;

	gamma_distribution<> gammaGen;

	double C;
};





#endif // ML_STUDENT_DISTRIBUTION_H