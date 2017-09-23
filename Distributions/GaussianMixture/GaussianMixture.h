#ifndef ML_GAUSSIAN_MIXTURE_DISTRIBUTION_H
#define ML_GAUSSIAN_MIXTURE_DISTRIBUTION_H

#include "../../Modelo.h"

#include "../Gaussian/Gaussian.h"
#include "../Multinomial/Multinomial.h"


struct GaussianMixture : public Multinomial
{
	GaussianMixture () {}

	template <class V, class... Gs>
	GaussianMixture (V&& alpha_, Gs&&... gs) : Multinomial(forward<V>(alpha_)), gaussians{forward<Gs>(gs)...} {}


	template <class V, class... Gs>
	void params (V&& alpha_, Gs&&... gs)
	{
		Multinomial::params(forward<V>(alpha_));
		gaussians = vector<Gaussian>{forward<Gs>(gs)...};
	}


	void update ()
	{
		Multinomial::update();

		for(auto& g : gaussians)
			g.update();
	}



	auto params ()
	{
		return make_tuple(Multinomial::params(), gaussians);
	}


	double operator () (const Vec& x)
	{
		double r = 0.0;

		for(int i = 0; i < K; ++i)
			r += Multinomial::operator()(i) * gaussians[i](x);

		return r;
	}

	Vec operator () ()
	{
		return gaussians[Multinomial::operator()()]();
	}



	void fit (const Mat& X, int K = 2, double precision = 1e-2)
	{
		int M = X.rows(), N = X.cols();


		Multinomial::params(Vec::Constant(K, 1.0 / K));

		gaussians = vector<Gaussian>(K, Gaussian(Vec(N), Mat::Constant(N, N, 0.0)));


		vi perm(M);
		iota(ALL(perm), 0);
		shuffle(ALL(perm), gen);

		for(int i = 0; i < K; ++i)
			gaussians[i].mu = X.row(perm[i]);


		// Vec l = X.colwise().minCoeff(), u = X.colwise().maxCoeff();

		// for(int i = 0; i < K; ++i)
		// 	for(int j = 0; j < N; ++j)
		// 		gaussians[i].mu(j) = l(j) + randD(0.0, 1.0) * (u(j) - l(j));



		Mat diff = X.rowwise() - X.colwise().mean();

		Mat var = Mat::Constant(N, N, 0.0);

		for(int i = 0; i < M; ++i)
			var += diff.row(i).transpose() * diff.row(i);

		var /= M;


		//DB(var << "\n\n");


		for(int i = 0; i < K; ++i)
		{
			gaussians[i].sigma = var;

			gaussians[i].update();
		}



		Mat R(M, K);

		double llOld, llNew = logLikelihood(X);



		// DB(mu << "\n");
		// DB(gaussians[0].mu << "\n");
		// DB(gaussians[1].mu << "\n\n\n\n");

		do
		{
			//DB(llNew << "\n\n\n");

			llOld = llNew;


			for(int i = 0; i < M; ++i)
			{
				for(int j = 0; j < K; ++j)
					R(i, j) = Multinomial::operator()(j) * gaussians[j](X.row(i));

				R.row(i) /= R.row(i).sum();
			}

			Vec colSum(K);

			for(int j = 0; j < K; ++j)
				colSum(j) = R.col(j).sum();
			


			Multinomial::mu.setZero();

			for(int j = 0; j < K; ++j)
				Multinomial::mu(j) += colSum(j);

			Multinomial::update();


			for(int j = 0; j < K; ++j)
			{
				gaussians[j].mu.setZero();
				gaussians[j].sigma.setZero();
			}


			for(int i = 0; i < M; ++i)
				for(int j = 0; j < K; ++j)
					gaussians[j].mu += R(i, j) * X.row(i);

			for(int j = 0; j < K; ++j)
				gaussians[j].mu /= colSum(j);


			for(int i = 0; i < M; ++i)
				for(int j = 0; j < K; ++j)
					gaussians[j].sigma += R(i, j) * (X.row(i).transpose() - gaussians[j].mu) * 
												    (X.row(i) - gaussians[j].mu.transpose());


			for(int j = 0; j < K; ++j)
			{
				gaussians[j].sigma /= R.col(j).sum();
				gaussians[j].update();
			}



			llNew = logLikelihood(X);


			// DB(mu << "\n");
			// DB(gaussians[0].mu << "\n");
			// DB(gaussians[1].mu << "\n\n\n\n\n");


		} while(llNew - llOld > precision);
	}



	double logLikelihood (const Mat& X)
	{
		double ll = 0.0;

		for(int i = 0; i < X.rows(); ++i)
		{
			double aux = 0.0;
			
			for(int j = 0; j < K; ++j)
				aux += Multinomial::operator()(j) * gaussians[j](X.row(i));

			ll += log(aux);
		}

		return ll;
	}



	vector<Gaussian> gaussians;

	RandDouble randD;
};





#endif // ML_GAUSSIAN_MIXTURE_DISTRIBUTION_H