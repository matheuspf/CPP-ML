#ifndef CPP_ML_OVA_H
#define CPP_ML_OVA_H

#include "../../Modelo.h"

#include "../ClassEncoder.h"


template <class Classifier, bool Encode = true>
struct OVA : public ClassEncoder<OVA<Classifier, Encode>, Encode>
{
public:

    using Base = ClassEncoder<OVA<Classifier, Encode>, Encode>;
    using Base::fit, Base::predict, Base::numClasses;


    template <typename... Args>
    OVA (Args&&... args) : classifierBase(std::forward<Args>(args)...) {}


    OVA& fit_ (const Mat& X, const Veci& y)
    {
        M = X.rows(), N = X.cols();

        classifiers.resize(numClasses, classifierBase);

        Veci yk(M);


        for(int k = 0; k < numClasses; ++k)
        {
            std::transform(std::begin(y), std::end(y), std::begin(yk), [&](int x)
            {
                return x == k ? 1 : 0;
            });

            classifiers[k].fit(X, yk);
        }

        return *this;
    }



    int predict_ (const Vec& x)
    {
        Vec classMargin(numClasses);

        std::transform(std::begin(classifiers), std::end(classifiers), std::begin(classMargin), [&](auto& cls)
        {
            return cls.predictMargin(x);
        });

        return std::max_element(std::begin(classMargin), std::end(classMargin)) - std::begin(classMargin);
    }


    Veci predict_ (const Mat& X)
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

    

    int M, N;
    
    std::vector<Classifier> classifiers;

    Classifier classifierBase;
};



#endif // CPP_ML_OVA_H