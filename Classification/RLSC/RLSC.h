#ifndef CPP_ML_RLSC_H
#define CPP_ML_RLSC_H

#include "../../Modelo.h"

#include "../../Kernels.h"

#include "../ClassEncoder.h"

#include "../../Preprocessing/Preprocess.h"



template <class Kernel = LinearKernel>
struct RLSC : public ClassEncoder<RLSC<Kernel>>
{
    RLSC (double alpha = 0.0, const Kernel& kernel = Kernel()) : alpha(alpha), kernel(kernel) {}

    RLSC (const Kernel& kernel, double alpha = 0.0) : alpha(alpha), kernel(kernel) {}


    void fit_ (const Mat& X, const Veci& y)
    {
        Z = X;

        Mat K = kernel(X, X);

        K.diagonal().array() += alpha;

        w = solveMat(K, y.cast<double>());
    }



    int predict_ (const Vec& x)
    {
        return predictMargin(x) > 0.0 ? 1 : -1;
    }

    Veci predict_ (const Mat& X)
    {
        Vec res = predictMargin(X);
        
        std::transform(std::begin(res), std::end(res), std::begin(res), [](double x){ return x > 0.0 ? 1.0 : -1.0; });

        return res.cast<int>();
    }


    double predictMargin (const Vec& x)
    {
        return w.dot(kernel(Z, x));
    }

    Vec predictMargin (const Mat& X)
    {
        return kernel(Z, X).transpose() * w;
    }



    int M, N;

    double alpha;

    Kernel kernel;
    
    Mat Z;

    Vec w;


    static std::vector<int> classLabels;
};


template <class Kernel>  
std::vector<int> RLSC<Kernel>::classLabels = {1, -1};



#endif // CPP_ML_RLSC_H