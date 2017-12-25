#ifndef CPP_ML_OVA_H
#define CPP_ML_OVA_H

#include "../../Modelo.h"

#include "../Classifier.h"



namespace impl
{

template <class Cls>
struct OVA : public Cls
{
public:

    USING_CLASSIFIER(Cls);
    using Cls::Cls;


    // template <typename... Args>
    // OVA (Args&&... args) : baseClassifier(std::forward<Args>(args)...) {}
    

    void fit (const Mat& X, const Veci& y)
    {
        M = X.rows(), N = X.cols();

        classifiers.resize(numClasses, *this);

        Veci yk(M);


        for(int k = 0; k < numClasses; ++k)
        {
            std::transform(std::begin(y), std::end(y), std::begin(yk), [&](int x)
            {
                return x == k ? positiveClass : negativeClass;
            });

            classifiers[k].fit(X, yk);
        }
    }



    int predict (const Vec& x)
    {
        Vec classMargin(numClasses);

        std::transform(std::begin(classifiers), std::end(classifiers), std::begin(classMargin), [&](auto& cls)
        {
            return cls.predictMargin(x);
        });

        return std::max_element(std::begin(classMargin), std::end(classMargin)) - std::begin(classMargin);
    }


    Veci predict (const Mat& X)
    {
        Mat classProb(X.rows(), numClasses);

        for(int k = 0; k < numClasses; ++k)
            classProb.col(k) = classifiers[k].predictMargin(X);

        Veci labels(X.rows());

        for(int i = 0; i < X.rows(); ++i)
        {
            double val = -1e20;
            
            for(int k = 0; k < numClasses; ++k)
            {
                if(classProb(i, k) > val)
                {
                    val = classProb(i, k);
                    labels(i) = k;
                }
            }
        }

        return labels;
    }


    Vec predictMargin (const Mat&) { return Vec(); }
    Vec predictProb (const Mat&) { return Vec(); }

    

    std::vector<Cls> classifiers;
};

}


template <class Cls, bool EncodeLabels = true>
using OVA = impl::Classifier<impl::OVA<Cls>, EncodeLabels>;


#endif // CPP_ML_OVA_H