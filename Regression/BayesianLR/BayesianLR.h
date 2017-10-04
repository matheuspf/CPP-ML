/** \file MDE.h
 * 
 *  This is an implementation of the Bayesian Linear Regression method.
 * 
 *  The hyperparameters are estimated based on an optimization over the evidence function,
 *  after integrating out the weights of the model.
 * 
 *  Basically, it is a regularized least squares, or 'Ridge' regression, without the need 
 *  for crossvalidation: the 'penalty' are automatically estimated.
 * 
 *  For the theory behind the method, see chapter 3 of Bishop's book.
 * 
*/


#ifndef ML_BAYESIAN_LR_H
#define ML_BAYESIAN_LR_H

#include "../../Modelo.h"



/// Stands for Bayesian Linear Regression
struct BayesianLR
{
	/// Nothing to be set here
	BayesianLR () {}

	/** The fit method also takes two optional parameters: the maximum number of iterations 
	 *  to optimize 'alpha' and 'beta', and the minimum tolerance between iterations.
	**/
	BayesianLR& fit (Mat X, const Vec& y, int maxIter = 100, double tol = 1e-8)
	{
		assert(X.rows() == y.rows() && "Different number of rows between X and y");

		/** This is needed, since we need a column full of 1's for the itercept term.
		  * Until the moment, this is the easiest and cleanest way I found to do that.
		  * Although it may not be the fastest way, the optimization iteration procedure, 
		  * which is O(N^3) per iteration, should dominate this O(M) memory allocation.
		**/
		X.conservativeResize(Eigen::NoChange, X.cols() + 1);
		X.col(X.cols() - 1).array() = 1.0;


		/// Number of rows (M) and columns (N)
		M = X.rows(), N = X.cols();


		/// Precomputed values
		Mat XtX = X.transpose() * X;

		Vec Xy = X.transpose() * y;


		/// We need the eigenvalues of XtX.
		EigenSolver<Mat> es(XtX);

		ArrayXd eigVal = es.eigenvalues().real().array();



		Mat In = Mat::Identity(N, N);
		
		/** Really important stuff here. We should not include a prior over the intercept.
		  * Actually, doing this, we include a prior, but it is infinitelly flat = uniform.
		  * This will not penalize any value chosen for the intercept term.
		**/
		In(N-1, N-1) = 0.0;


		/// For convergence check
		double oldAlpha = 1e20, oldBeta = 1e20;

		/// Initial values
		alpha = beta = 1.0;


		/// While there's still change in 'alpha' or 'beta'
		while((abs(alpha - oldAlpha) > tol || abs(beta - oldBeta) > tol) && maxIter--)
		{
			/// Old values
			oldAlpha = alpha, oldBeta = beta;

			/** We will need this matrix later to evaluate the conditional distribution P(y, x).
			 *  The 'invertMat' tries to apply a Cholesky LLT decomposition. If it does not work,
			 *  a QR decomposition is applied. If M >> N, it generally wont be necessary.
			**/
			sigma = inverseMat(beta * XtX + alpha * In);

			/// This is our weight vector
			mu = beta * sigma * Xy;

			/// This guy represents the 'effective number of parameters'. See Bishop's book.
			double gamma = ((beta * eigVal) / (alpha + beta * eigVal)).sum();

			/// New values
			alpha = gamma / (mu.dot(mu));

			beta = (N - gamma) / (y - X * mu).squaredNorm();
		}

		/// The intercept is the last term
		intercept = mu(N-1);

		/// Resize so inference is a simply dot plus an add
		mu.conservativeResize(N-1);


		return *this;
	}



	/// Predicting values for a vector 'x'
	double predict (const Vec& x)
	{
		return mu.dot(x) + intercept;
	}

	/// Predicting values for a whole matrix 'X'
	Vec predict (const Mat& X)
	{
		return (X * mu).array() + intercept;
	}



	int M, N;

	double alpha;

	double beta;
	
	Mat sigma;

	Vec mu;

	double intercept;

};



#endif // ML_BAYESIAN_LR_H