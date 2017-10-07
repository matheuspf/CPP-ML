/** \file BayesianDualLR.h
 * 
 *  Code for Bayesian Dual Linear Regression.
 * 
 *  Based on the Bayesian Linear Regression derivation in chapter 3 of Bishop's book.
**/

#ifndef CPP_ML_BAYESIAN_DUAL_LR_H
#define CPP_ML_BAYESIAN_DUAL_LR_H

#include "../../Modelo.h"

#include "../../Kernels.h"            /// The kernel functions

#include "../../SpectraHelpers.h"     /// Spectra for calculating the eigenvalues/eigenvectors of large matrices



/// Linear kernel as default - includes an additional sum with a constant
template <class Kernel = LinearKernel>
struct BayesianDualLR
{
    /// If you want to pass kernel parameters. The class itself has no parameter to be set
    BayesianDualLR (const Kernel& kernel = Kernel()) : kernel(kernel) {}

    
    /// 'maxIter' for the maximum number of iterations in the optimization, and 'tol', for the change tolerance
    BayesianDualLR& fit (const Mat& X, const Vec& y, int maxIter = 10, double tol = 1e-8)
    {
        /// Assert if X and y have the same number of rows
        assert(X.rows() == y.rows() && "Number of dimensions between 'X' and 'y' differ");

        M = X.rows(), N = X.cols();     /// Number of rows and columns
        
        Z = X;      /// We need to save the matrix for further prediction


        Mat K = kernel(X, X);     /// Calculate the kernel matrix

        Vec Ky = K * y;           /// And an additional precomputed vector

        /** This is a utility function that uses Spectra to find the N+1 largest positive eigenvalues.
         *  Notice: N+1 eigenvalues because we are simulating a entire column of ones in the kernel.
        **/ 
        TopEigen<Spectra::SELECT_EIGENVALUE::LARGEST_ALGE> topEigen(K, N+1); 

        ArrayXd eigVals = topEigen.eigenvalues().array();     /// Retrieve only the eigenvalues. They are sorted

        
        /// For convergence check
        double oldAlpha = 1e20, oldBeta = 1e20;
        
        /// Initial values
        alpha = beta = 1.0;


		/// While there's still change in 'alpha' or 'beta'
		while((abs(alpha - oldAlpha) > tol || abs(beta - oldBeta) > tol) && maxIter--)
		{
			/// Old values
			oldAlpha = alpha, oldBeta = beta;


            /** This matrix is need for later conditional probability evaluation of P(y | x).
             *  In general, when M > N, we will need a QR decomposition (see 'inverseMat').
            **/
            sigma = beta * K * K;
            
            sigma.diagonal().array() += alpha;

            sigma = inverseMat(sigma);


            /// The actual weights
            mu = beta * sigma * Ky;


            /// The 'number of effective parameters'
			double gamma = ((beta * eigVals) / (alpha + beta * eigVals)).sum();


            /// Calculate the new values
			alpha = gamma / (mu.dot(mu));

            beta = (N - gamma) / (y - K * mu).squaredNorm();
        }


        return *this;      /// Return a reference to this
    }


    /// Prediction is accomplished through a kernel transformation plus a dot product with the weights
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

    Mat sigma;          /// Matrix that incorporates data dependent and data independent variance

    Vec mu;             /// The actual weights

    Mat Z;              /// We have to keep the matrix we have trained on.
};



#endif // CPP_ML_BAYESIAN_DUAL_LR_H