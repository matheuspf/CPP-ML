#ifndef CPP_ML_BAYESIAN_DUAL_LR_H
#define CPP_ML_BAYESIAN_DUAL_LR_H

#include "../../Modelo.h"

#include "../../Kernels.h"

#include "../../SpectraHelpers.h"


template <class Kernel = LinearKernel>
struct BayesianDualLR
{
    BayesianDualLR (const Kernel& kernel = Kernel()) : kernel(kernel) {}

    
    BayesianDualLR& fit (const Mat& X, const Vec& y, int maxIter = 100, double tol = 1e-8)
    {
        /// Assert if X and y have the same number of rows
        assert(X.rows() == y.rows() && "Number of dimensions between 'X' and 'y' differ");

        M = X.rows(), N = X.cols();     /// Number of rows and columns
        
        Z = X;


        Mat K = kernel(X, X);

        Vec Ky = K * y;

        TopEigen<Spectra::SELECT_EIGENVALUE::LARGEST_ALGE> topEigen(K, N);

        ArrayXd eigVals = topEigen.eigenvalues().array();


        /// For convergence check
        double oldAlpha = 1e20, oldBeta = 1e20;
        
        /// Initial values
        alpha = beta = 1.0;


		/// While there's still change in 'alpha' or 'beta'
		while((abs(alpha - oldAlpha) > tol || abs(beta - oldBeta) > tol) && maxIter--)
		{
			/// Old values
			oldAlpha = alpha, oldBeta = beta;


            sigma = beta * K * K;
            
            sigma.diagonal().array() += alpha;

            sigma = inverseMat(sigma);

            mu = beta * sigma * Ky;


			double gamma = ((beta * eigVals) / (alpha + beta * eigVals)).sum();


			alpha = gamma / (mu.dot(mu));

            beta = (N - gamma) / (y - K * mu).squaredNorm();


            db(alpha, "      ", beta);
        }


        return *this;
    }



    double predict (const Vec& x) const
    {
        return mu.dot(kernel(Z, x));
    }

    Vec predict (const Mat& X) const
    {
        return kernel(Z, X).transpose() * mu;
    }




    int M, N;           /// Dimensions of the matrix (M - rows, N - columns)

	double alpha;		/// Inverse of the variance of the isotropic Gaussian prior over the weights
    
    double beta;		/// Inverse of the variance of the Gaussian conditional distribution P(y | x)

    Kernel kernel;      /// Kernel function

    Mat sigma;

    Vec mu;            /// The actual weights

    Mat Z;              /// We have to keep the matrix we have trained on.
};



#endif // CPP_ML_BAYESIAN_DUAL_LR_H