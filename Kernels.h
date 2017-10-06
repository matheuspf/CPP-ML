#ifndef CPP_ML_KERNELS_H
#define CPP_ML_KERNELS_H

#include "Modelo.h"


struct LinearKernel
{
	LinearKernel (double c = 1.0) : c(c) {}

	double operator () (const Vec& x, const Vec& y) const
	{
		return x.dot(y) + c;
	}

	Vec operator () (const Mat& X, const Vec& y) const
	{
		return (X * y).array() + c;
	}

	Mat operator () (const Mat& X, const Mat& Z) const
	{
		return (X * Z.transpose()).array() + c;
	}

	double c;
};


struct RBFKernel
{
	RBFKernel (double gamma = 0.1) : gamma(gamma) {}

	double operator () (const Vec& x, const Vec& y) const
	{
		return exp(-gamma * (x - y).squaredNorm());
	}

	Vec operator () (const Mat& X, const Vec& y) const
	{
		return exp(-(gamma * (X.rowwise() - y.transpose()).rowwise().squaredNorm()).array());
	}


	Mat operator () (const Mat& X, const Mat& Z) const
	{
		Mat res(X.rows(), Z.rows());

		for(int i = 0; i < X.rows(); ++i)
			for(int j = 0; j < Z.rows(); ++j)
				res(i, j) = this->operator()(Vec(X.row(i)), Vec(Z.row(j)));

		return res;
	}


	double gamma;
};


#endif // CPP_ML_KERNELS_H