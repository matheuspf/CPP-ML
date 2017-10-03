#ifndef CPP_ML_DUAL_LR_H
#define CPP_ML_DUAL_LR_H

#include "../../Modelo.h"

#include "../../Kernels.h"


template <class Kernel = LinearKernel>
struct DualLR
{
    DualLR (double alpha = 0.0, double alphaBias = 0.0, const Kernel& kernel = Kernel()) :
             alpha(alpha), alphaBias(alphaBias), kernel(kernel) {}

    
    DualLR& fit (Mat X, const Vec& y)
    {
        assert(X.rows() == y.rows() && "Number of dimensions between 'X' and 'y' differ");

        X.conservativeResize(Eigen::NoChange, X.cols()+1);
        X.col(X.cols()-1).array() = 1.0;


        int M = X.rows(), N = X.cols() - 1;
        

        Mat Im = Mat::Identity(M, M);
        
        Im(M-1, M-1) = (alpha == 0.0 ? 0.0 : alphaBias / alpha);


        phi = (kernel(X, X.transpose()) + alpha * Im).colPivHouseholderQr().solve(y);


        Z = X;

        return *this;
    }


    double predict (Vec x) const
    {
        x.conservativeResize(x.rows() + 1);
        x(x.rows()-1) = 1.0;

        return phi.dot(kernel(Z, x));
    }

    // Vec predict (Mat X) const
    // {
    //     X.conservativeResize(Eigen::NoChange, X.cols()+1);
    //     X.col(X.cols()-1).array() = 1.0;

    //     return kernel(Z.transpose(), X) * phi;
    // }


    double alpha, alphaBias;

    Kernel kernel;

    Vec phi;

    double bias;

    Mat Z;
};


#endif // ML_DUAL_LR_H