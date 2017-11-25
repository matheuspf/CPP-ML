#ifndef CPP_ML_UNSUPERVISED_PCA_H
#define CPP_ML_UNSUPERVISED_PCA_H

#include "../../Modelo.h"
#include "../../Preprocessing/Preprocess.h"
#include "../../SpectraHelpers.h"


struct PCA
{
    PCA (int D = 2) : D(D)
    {
        assert(D >= 1 && "The projected dimension must have dimension of at least one");
    }


    PCA& fit (Mat X)
    {
        int M = X.rows(), N = X.cols();

        xMean = X.colwise().mean();

        X = X.rowwise() - xMean.transpose();

        Mat S;


        if(N > M)
            S = (1.0 / N) * X * X.transpose();
    
        else
            S = (1.0 / N) * X.transpose() * X;

            
        TopEigen<> largestEigen(S, std::min(D, std::min(N, M)));

        U = largestEigen.eigenvectors();

        L = largestEigen.eigenvalues();


        if(N > M)
            U = X.transpose() * U;


        U = U.array().rowwise() / U.colwise().norm().array();


        return *this;
    }


    Mat transform (const Mat& X)
    {
        return X * U;
    }

    Vec transform (const Vec& x)
    {
        return x * U;
    }


    auto fitTransform (const Mat& X)
    {
        return fit(X).transform(X);
    }


    int D;

    Mat U;

    Vec L;

    Vec xMean;

};



struct Whitening : public PCA
{
    using PCA::PCA;

    
    Whitening& fit (const Mat& X)
    {
        PCA::fit(X);

        LInv = Eigen::sqrt(1.0 / L.array());

        return *this;
    }


    Mat transform (const Mat& X)
    {
        return (X.rowwise() - xMean.transpose()) * U * LInv.asDiagonal();
    }


    auto fitTransform (const Mat& X)
    {
        return fit(X).transform(X);
    }


    Vec LInv;
};




#endif // CPP_ML_UNSUPERVISED_PCA_H