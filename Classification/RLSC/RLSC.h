#ifndef CPP_ML_RLSC_H
#define CPP_ML_RLSC_H

#include "../../Modelo.h"

#include "../../Kernels.h"

#include "../../Preprocessing/Preprocess.h"


template <class Kernel = LinearKernel>
struct RLSC
{
    RLSC (double alpha = 0.0, const Kernel& kernel = Kernel()) : alpha(alpha), kernel(kernel) {}

    RLSC (const Kernel& kernel, double alpha = 0.0) : alpha(alpha), kernel(kernel) {}


    RLSC& fit (const Mat& X, Veci y, bool preProcessLabels_ = true)
    {
        assert(X.rows() == y.rows() && "Observation matrix and labels differ in number of samples.");

        M = X.rows(), N = X.cols();
        preProcessLabels = preProcessLabels_;

        posClass = Vec::Constant(M, 1.0), negClass = -posClass;

        Z = X;

        if(preProcessLabels)
        {
            lenc = LabelEncoder<int>();
            y = lenc.fitTransform(y, {-1, 1});
        }



        Mat K = kernel(X, X);

        K.diagonal().array() += alpha;

        w = solveMat(K, y.cast<double>());


        return *this;
    }



    int predict (const Vec& x)
    {
        return predictMargin(x) > 0.0 ? 1 : -1;
    }

    Veci predict (const Mat& X)
    {
        Vec res = predictMargin(X);
        
        std::transform(std::begin(res), std::end(res), std::begin(res), [](double x){ return x > 0.0 ? 1.0 : -1.0; });

        if(preProcessLabels)
        {
            Veci y(res.rows());

            for(int i = 0; i < res.rows(); ++i)
                y(i) = lenc.reverseMap[int(res(i))];
            
            return y;
        }

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

    Vec posClass, negClass;


    LabelEncoder<int> lenc;

    bool preProcessLabels;
};



#endif // CPP_ML_RLSC_H