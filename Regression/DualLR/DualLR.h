/** \file DualLR.h
 *  
 *  Implementation of Dual Linear Regression.
 *  
 *  In the Dual Linear Regression, the weights are represented as a linear combination
 *  of the rows (observations) of the observation matrix X. That is,
 * 
 *  \f$ w = X^Ta \f$
 * 
 *  Where w is the same weight vector used in the ordinary linear regression, and a is 
 *  the M dimensional vector whose values we have to find.
**/

#ifndef CPP_ML_DUAL_LR_H
#define CPP_ML_DUAL_LR_H

#include "../../Modelo.h"


/// The kernel functions
#include "../../Kernels.h"



/** By default, the kernel is linear (simple dot product). Notice here that the 
 *  LinearKernel also adds constant, so we don't need to add a whole new column
 *  to the observation matrix or to every new element to predict. Of course, you
 *  can set the constant to 0 and obtain the original kernel.
**/
template <class Kernel = LinearKernel>
struct DualLR
{
    /// No regularization by default (alpha = 0)
    DualLR (double alpha = 0.0, const Kernel& kernel = Kernel()) : alpha(alpha), kernel(kernel) {}

    
    DualLR& fit (const Mat& X, const Vec& y)
    {
        /// Assert if X and y have the same number of rows
        assert(X.rows() == y.rows() && "Number of dimensions between 'X' and 'y' differ");

        M = X.rows(), N = X.cols();     /// Number of rows and columns
        
        /** Here we calculate the kernel matrix K, where \f$ K_{i, j} = kernel(X_i, X_j) \f$
         *  and add the regularizer on the diagonal. Then, we solve for y, using the 'solveMat'
         *  function, that attemps to use a Cholesky LLT decomposition. It will not work if
         *  M > N, so a QR decomposition is used instead.
        **/
        Mat K = kernel(X, X);

        K.diagonal().array() += alpha;

        phi = solveMat(K, y);


        /// We have to keep the observation matrix for prediction
        Z = X;

        return *this;
    }



    /// Use kernel function with the saved observation matrix and then a dot product with the weights.
    double predict (const Vec& x) const
    {
        return phi.dot(kernel(Z, x));
    }

    Vec predict (const Mat& X) const
    {
        return kernel(Z, X).transpose() * phi;
    }


    int M, N;           /// Dimensions of the matrix (M - rows, N - columns)

    double alpha;       /// Regularization constant - inverse of the variance of the gaussian prior over weights

    Kernel kernel;      /// Kernel function

    Vec phi;            /// The actual weights

    Mat Z;              /// We have to keep the matrix we have trained on.
};


#endif // ML_DUAL_LR_H