#ifndef ML_SPARSE_BAYESIAN_LR_H
#define ML_SPARSE_BAYESIAN_LR_H

#include "../../Modelo.h"


struct SparseBayesianLR
{
    SparseBayesianLR (double alphaA = 0.0, double alphaB = 0.0, double betaA = 0.0, double betaB = 0.0) :
                      alphaA(alphaA), alphaB(alphaB), betaA(betaA), betaB(betaB) {}



	SparseBayesianLR& fit (Mat X, const Vec& y, double tol = 1e-3, int maxIter = 100)
	{
        X.conservativeResize(Eigen::NoChange, X.cols()+1);
		X.col(X.cols()-1).array() = 1.0;

        M = X.rows(), N = X.cols();

        alphas = Vec::Constant(N, 1.0);
        beta = 1.0;

        indices = vector<int>(N);
        iota(indices.begin(), indices.end(), 0);

        do
        {
            sigma = (beta * X.transpose() * X + Mat(alphas.asDiagonal())).inverse();
            
            mu = beta * sigma * X.transpose() * y;

            beta = 0.0;

            for(int i = 0; i < N; ++i)
            {
                double gamma = 1.0 - alphas(i) * sigma(i, i);

                beta += gamma;

                alphas(i) = (gamma + alphaA) / (pow(mu(i), 2) + alphaB);
            }

            beta = (N - beta + alphaA) / ((y - X * mu).squaredNorm() + betaB);


            // for(int i = 0; i < N; ++i) if(alphas(i) > 1e3)
            // {
            //     N--;

            //     if(i != N)
            //     {
            //         X.block(0, i, M, N-i) = X.rightCols(N-i);
            //         alphas.tail(N-i) = alphas.tail(N-i-1);
            //     }

            //     X.conservativeResize(Eigen::NoChange, N);
            //     alphas.conservativeResize(N);
            //     indices.erase(indices.begin() + i);
            // }

        } while(--maxIter);

        db(alphas.transpose(), "\n\n");
    }
    

    double operator () (Vec x)
    {
        x.conservativeResize(x.rows() + 1);
        x(x.rows() - 1) = 1.0;
        
        return mu.dot(x);
    }


    Vec alphas;
    double beta;

    vector<int> indices;

    double alphaA, alphaB;  /// Hyperpriors of Gamma distributions over the priors.
    double betaA, betaB;    /// If set to 0, they are uniform.

    int M, N;
    
    Vec mu;
    Mat sigma;
};



#endif // ML_SPARSE_BAYESIAN_LR_H