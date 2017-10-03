#ifndef CPP_ML_KERNELS_H
#define CPP_ML_KERNELS_H

#include "Modelo.h"


struct LinearKernel
{
	template <class X, class Y>
	auto operator () (const X& x, const Y& y) const
	{
		return x * y;
	}
};


struct RBFKernel
{
	RBFKernel (double gamma = 0.1) : gamma(gamma) {}

	double operator () (const Vec& x, const Vec& y) const
	{
		return exp(-(x - y).squaredNorm() * gamma);
	}

	Vec operator () (const Mat& X, const Vec& y) const
	{
		return exp(-((X.rowwise() - y.transpose()).rowwise().squaredNorm() * gamma).array());
	}


	Mat operator () (const Mat& X) const
	{
		Mat res(X.rows(), X.rows());

		for(int i = 0; i < X.rows(); ++i)
			for(int j = 0; j < X.rows(); ++j)
				res(i, j) = this->operator()(Vec(X.row(i)), Vec(X.row(j)));

		return res;
	}


	// Mat operator () (const Mat& X, const Mat& Y) const
	// {
	// 	Mat res(X.rows(), Y.cols());

	// 	for(int i = 0; i < X.rows(); ++i)
	// 		res.col(i) = this->operator()(X, Vec(Y.row(i)));

	// 	return res;
	// }


	double gamma;
};


#endif // CPP_ML_KERNELS_H