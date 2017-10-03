#ifndef ML_SPARSE_BAYESIAN_LR_H
#define ML_SPARSE_BAYESIAN_LR_H

#include "../../Modelo.h"


struct SparseBayesianLR
{
    SparseBayesianLR (double alphaA = 0.0, double alphaB = 0.0, double betaA = 0.0, double betaB = 0.0) :
                      alphaA(alphaA), alphaB(alphaB), betaA(betaA), betaB(betaB) {}



	SparseBayesianLR& fit (Mat X, const Vec& y, double tol = 1e-3, int maxIter = 10)
	{
        X.conservativeResize(Eigen::NoChange, X.cols()+1);
        X.col(X.cols()-1).array() = 1.0;
        
        Mat XtX = X.transpose() * X;

        M = X.rows(), N = X.cols();

        alphas = Vec::Constant(N, 1.0);

        //alphas(N-1) = 0.0;
        beta = 1.0;

        indices = vector<int>(N);
        iota(indices.begin(), indices.end(), 0);


        EigenSolver<Mat> es(XtX);

        ArrayXd eigVals = es.eigenvalues().real().array();

        Vec gammas(N);

        Vec oldAlphas;

        do
        {
            oldAlphas = alphas;

            sigma = (beta * XtX + Mat(alphas.asDiagonal())).inverse();
            
            mu = beta * sigma * X.transpose() * y;

            gammas = (beta * eigVals) / (alphas.array() + beta * eigVals);

            alphas = (gammas.array() + alphaA) / (pow(mu.array(), 2) + alphaB);

            //alphas(N-1) = 0.0;
            
            beta = (N - gammas.sum() + betaA) / ((y - X * mu).dot(y - X * mu) + betaB);


            db(beta, "     ", alphas.transpose(), "\n");

            // if((alphas - oldAlphas).norm() / alphas.norm() < tol*N)
            //     break;


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

        db("\n\n", mu.transpose(), "\n\n\n");

        //db(maxIter, "     ", beta, "     ", alphas.transpose(), "\n\n");
    }
    

    double predict (Vec x)
    {
        x.conservativeResize(x.rows() + 1);
        x(x.rows() - 1) = 1.0;
        
        return mu.dot(x);
    }

    Vec predict (Mat X)
    {
        X.conservativeResize(Eigen::NoChange, X.cols()+1);
        X.col(X.cols()-1).array() = 1.0;
        
        return X * mu;
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