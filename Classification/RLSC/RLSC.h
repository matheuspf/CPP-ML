#ifndef CPP_ML_RLSC_H
#define CPP_ML_RLSC_H

#include "../../Modelo.h"

#include "../../Kernels.h"

#include "../Classifier.h"

#include "../../Preprocessing/Preprocess.h"


namespace impl
{

template <class Kernel = LinearKernel, bool Polymorphic = false>
struct RLSC : public PickClassifier<Polymorphic>
{
    USING_CLASSIFIER(PickClassifier<Polymorphic>);


    RLSC (double alpha = 0.0, const Kernel& kernel = Kernel()) :
          BaseClassifier(1, -1, false), alpha(alpha), kernel(kernel) {}

    RLSC (const Kernel& kernel, double alpha = 0.0) : 
          BaseClassifier(1, -1, false), alpha(alpha), kernel(kernel) {}



    void fit (const Mat& X, const Veci& y)
    {
        Z = X;

        Mat K = kernel(X, X);

        K.diagonal().array() += alpha;

        w = solveMat(K, y.cast<double>());
    }



    int predict (const Vec& x)
    {
        return predictMargin(x) > 0.0 ? positiveClass : negativeClass;
    }

    Veci predict (const Mat& X)
    {
        Vec res = predictMargin(X);
        
        std::transform(std::begin(res), std::end(res), std::begin(res), [&](double x)
        {
            return x > 0.0 ? positiveClass : negativeClass;
        });

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



    double alpha;

    Kernel kernel;
    
    Mat Z;

    Vec w;
};

} // namespace impl


template <class Kernel = LinearKernel, bool EncodeLabels = true>
using RLSC = impl::Classifier<impl::RLSC<Kernel, false>, EncodeLabels>;


namespace poly
{

template <class Kernel = LinearKernel, bool EncodeLabels = true>
using RLSC = impl::Classifier<impl::RLSC<Kernel, true>, EncodeLabels>;

}



#endif // CPP_ML_RLSC_H