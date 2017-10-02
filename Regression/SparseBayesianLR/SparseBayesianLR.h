#ifndef ML_SPARSE_BAYESIAN_LR_H
#define ML_SPARSE_BAYESIAN_LR_H

#include "../../Modelo.h"


struct SparseBayesianLR
{
    SparseBayesianLR (double alphaA = 0.0, double alphaB = 0.0, double betaA = 0.0, double betaB = 0.0) :
                      alphaA(alphaA), alphaB(alphaB), betaA(betaA), betaB(betaB) {}



	SparseBayesianLR& fit (Mat X, const Vec& y, double tol = 1e-3, int maxIter = 1000)
	{
        X.conservativeResize(Eigen::NoChange, X.cols()+1);
        X.col(X.cols()-1).array() = 1.0;
        
        Mat XtX = X.transpose() * X;

        M = X.rows(), N = X.cols();

        alphas = Vec::Constant(N, 1.0); //alphas(N-1) = 1.0;
        beta = 1.0;

        indices = vector<int>(N);
        iota(indices.begin(), indices.end(), 0);

        Vec oldAlphas;

        do
        {
            oldAlphas = alphas;

            sigma = (beta * XtX + Mat(alphas.asDiagonal())).inverse();
            
            mu = beta * sigma * X.transpose() * y;

            beta = 0.0;

            for(int i = 0; i < N; ++i)
            {
                double gamma = 1.0 - alphas(i) * sigma(i, i);

                beta += gamma;

                alphas(i) = (gamma + alphaA) / (pow(mu(i), 2) + alphaB);
            }

            //alphas(N-1) = 1.0;
            
            beta = (N - beta + betaA) / ((y - X * mu).squaredNorm() + betaB);

            //db(beta, "     ", alphas.transpose(), "\n");

            if((alphas - oldAlphas).norm() < tol*N)
                break;


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


        sigma = (beta * XtX + Mat(alphas.asDiagonal())).inverse();
        
        mu = beta * sigma * X.transpose() * y;

        db(beta, "     ", alphas.transpose(), "\n\n");
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