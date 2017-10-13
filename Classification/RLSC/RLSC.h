#ifndef CPP_ML_RLSC_H
#define CPP_ML_RLSC_H

#include "../../Modelo.h"

#include "../../Kernels.h"


template <class Kernel = LinearKernel>
struct RLSC
{
    RLSC (double alpha = 0.0, const Kernel& kernel = Kernel()) : alpha(alpha), kernel(kernel) {}

    RLSC (const Kernel& kernel, double alpha = 0.0) : alpha(alpha), kernel(kernel) {}


    RLSC& fit (const Mat& X, const Veci& y)
    {
        assert(X.rows() == y.rows() && "Observation matrix and labels differ in number of samples.");

        M = X.rows(), N = X.cols();

        posClass = Vec::Constant(M, 1.0), negClass = -posClass;

        Z = X;


        Mat K = kernel(X, X);

        K.diagonal().array() += alpha;

        w = solveMat(K, y.cast<double>());


        return *this;
    }



    int predict (const Vec& x)
    {
        return w.dot(kernel(Z, x)) > 0.0 ? 1 : -1;
    }

    Veci predict (const Mat& X)
    {
        Vec res = kernel(Z, X).transpose() * w;
        
        std::transform(std::begin(res), std::end(res), std::begin(res), [](double x){ return x > 0.0 ? 1.0 : -1.0; });

        return res.cast<int>();
    }


    int M, N;

    double alpha;

    Kernel kernel;
    
    Mat Z;

    Vec w;

    Vec posClass, negClass;

};



#endif // CPP_ML_RLSC_H