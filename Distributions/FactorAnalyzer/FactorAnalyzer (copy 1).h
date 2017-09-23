#ifndef ML_FACTOR_ANALYZER_DISTRIBUTION_H
#define ML_FACTOR_ANALYZER_DISTRIBUTION_H

#include "../../Modelo.h"

#include "../Gaussian/Gaussian.h"

#include "../../Optimization/LineSearch/Bracketing/Bracketing.h"

#include "../../Optimization/LineSearch/Brents/Brents.h"

#include <boost/math/special_functions/digamma.hpp>



struct FactorAnalyzer : public Gaussian
{
	using Gaussian::Gaussian;
	using Gaussian::params;



	FactorAnalyzer (const Vec& mu_, const Mat& phi_, const Mat& var_)
	{
		params(mu_, phi_, var_);
	}


	void params (const Vec& mu_, const Mat& phi_, const Mat& var_)
	{
		mu = mu_;
		phi = phi_;
		var = var_;

		//Gaussian::params(mu_, (phi * phi.transpose()).colwise() + var);

		update();
	}

	void update ()
	{
		K = phi.cols();

		sigma = phi * phi.transpose() + var;

		Gaussian::update();
	}

	auto params ()
	{
		return make_tuple(mu, phi, var);
	}





	void fit (Mat X, int K_ = 2, double precision = 1e-2, int maxIter = 10)
	{
		int M = X.rows(), N = X.cols(); K = K_;


		mu = X.colwise().mean();

		phi = Mat::NullaryExpr(N, K, [&](int, int){ return rng(gen); });
		//phi = Mat::Identity(N, K);

		Vec pqp = (1.0 / M) * pow((X.rowwise() - mu.transpose()).array(), 2).matrix().colwise().sum();

		X = X.rowwise() - mu.transpose();


		Mat invVar = Mat::Constant(N, N, 0.0);

		var = Mat::Constant(N, N, 0.0);

		for(int i = 0; i < N; ++i)
			var(i, i) = pqp(i);


		vector<Vec> eh(M);
		Vec ehSum = Vec::Constant(K, 0.0);

		Mat EH(K, K);

		Mat phiInvA(K, K), phiInvB(K, N);
		Mat Ik = Mat::Identity(K, K);


		double ll = -1e13, oldLL;


		do
		{
			oldLL = ll;

			for(int i = 0; i < N; ++i)
				invVar(i, i) = 1.0 / var(i, i);

			phiInvA = (phi.transpose() * invVar * phi + Ik).inverse();

			phiInvB = phiInvA * phi.transpose() * invVar;



			EH.setZero();

			for(int i = 0; i < M; ++i)
			{
				eh[i] = phiInvB * X.row(i).transpose();

				EH += phiInvA + eh[i] * eh[i].transpose();
			}

			


			phi.setZero();

			for(int i = 0; i < M; ++i)
				phi += X.row(i).transpose() * eh[i].transpose();
			
			phi = phi * EH.inverse();




			Vec aux = Vec::Constant(N, 0.0);

			var.setZero();

			for(int i = 0; i < M; ++i)
			{
				var += (X.row(i).transpose() * X.row(i) - phi * (eh[i] * X.row(i)));


				// Vec diff = X.row(i).transpose() - mu;

				// aux += (pow(diff.array(), 2) + (phi * eh[i]).array() * diff.array()).matrix();
			}

			// for(int i = 0; i < N; ++i)
			// 	var(i, i) = aux(N-i-1);

			// var = (1.0 / M) * var;

			for(int i = 0; i < N; ++i) for(int j = 0; j < N; ++j)
				var(i, j) = (i == j) ? var(i, j) / M : 0.0;

			// for(int i = 0; i < N / 2; ++i)
			// 	swap(var(i, i), var(N-i-1, N-i-1));


			update();


			//ll = logLikelihood(X);

			//DB(maxIter << "    " << ll);

		
		} while(--maxIter);
		//} while(abs(ll - oldLL) > precision && --maxIter);
	}




	Mat phi;

	Mat var;

	int K;
};





#endif // ML_FACTOR_ANALYZER_DISTRIBUTION_H