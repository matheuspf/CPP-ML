#include "GaussianMixture.h"

#include "MDE/MDE.h"



auto getData ()
{
	int K = 3;

	Vec alphas(K);

	alphas(0) = 0.3;
	alphas(1) = 0.3;
	alphas(2) = 0.4;

	GaussianMixture dist(alphas,
					     Gaussian(Vec::Constant(2, -2.2), Mat::Identity(2, 2)),
					     Gaussian(Vec::Constant(2,  2.2), Mat::Identity(2, 2)),
					     Gaussian(Vec::Constant(2,  0.0), Mat::Identity(2, 2)));


	// Generate and fit

	int pts = 1e3;

	Mat X(pts, 2);

	FOR(i, pts)
		X.row(i) = dist();


	return make_tuple(X, K);
}



double EM (const Mat& X, int K)
{
	GaussianMixture gm;

	gm.fit(X, K, 1e-4);

	return gm.logLikelihood(X);
}





struct GMFunc : mde::Function<>
{
	GMFunc (const Mat& X, int K) : mde::Function<>(int(K + K * (X.cols() +  X.cols() * X.cols()))),
								   X(X), n(X.cols()), K(K)
	{
		Vec l = X.colwise().minCoeff(), u = X.colwise().maxCoeff();


		for(int i = 0; i < K; ++i)
			lowerBounds[i] = 0.0, upperBounds[i] = 1.0;

		for(int i = K, j = 0; i < K + K * n; ++i, j = (j + 1) % n)
			lowerBounds[i] = l(j), upperBounds[i] = u(j);


		for(int i = K + K * n; i < N; ++i)
			//lowerBounds[i] = l(i % K), upperBounds[i] = u(i % K);
			lowerBounds[i] = -5, upperBounds[i] = 5;



		gm.mu = Vec::Constant(K, 0.0);
		gm.gaussians = vector<Gaussian>(K);

		for(int i = 0; i < K; ++i)
		{
			gm.gaussians[i].mu = Vec::Constant(n, 0.0);
			gm.gaussians[i].sigma = Mat::Constant(n, n, 0.0);
		}
	}


	double operator () (const Vector& x)
	{
		copy(x.begin(), x.begin() + K, gm.mu.data());

		for(int i = 0; i < K; ++i)
			copy(x.begin() + K + i * n, x.begin() + K + (i + 1) * n, gm.gaussians[i].mu.data());

		for(int i = 0; i < K; ++i)
		{
			copy(x.begin() + K + K * n + i * (n * n), x.begin() + K + K * n + (i + 1) * (n * n), gm.gaussians[i].sigma.data());

			gm.gaussians[i].sigma = gm.gaussians[i].sigma.transpose() * gm.gaussians[i].sigma;
		}


		gm.update();

		return -gm.logLikelihood(X);
	}


	double inequalities (const Vector& x)
	{
		double g = 0.0;

		for(int k = 0; k < K; ++k)
		{
			copy(x.begin() + K + K * n + k * (n * n), x.begin() + K + K * n + (k + 1) * (n * n), gm.gaussians[k].sigma.data());

			gm.gaussians[k].sigma = gm.gaussians[k].sigma.transpose() * gm.gaussians[k].sigma;


			for(int i = 0; i < n; ++i)
				for(int j = i + 1; j < n; ++j)
					g += max(0.0, 2 * gm.gaussians[k].sigma(i, j) - gm.gaussians[k].sigma(i, i) - gm.gaussians[k].sigma(j, j));
		}

		return g;
	}



	Mat X;

	int n, K;


	GaussianMixture gm;
};




double DE (const Mat& X, int K)
{
	mde::Parameters params;

	params.maxIter = 200;


	mde::MDE<GMFunc> mde(params, GMFunc(X, K));

	mde();

	return -mde.function(mde.best);
}





int main ()
{
	Mat X;

	int K = 3;

	tie(X, K) = getData();


	DB(EM(X, K));

	DB(DE(X, K));



	return 0;
}