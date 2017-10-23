#ifndef CPP_ML_OVA_H
#define CPP_ML_OVA_H

#include "../../Modelo.h"

#include "../ClassEncoder.h"


template <class Classifier>
struct OVA : public Classifier, ClassEncoder<OVA<Classifier>>
{
public:

    using Classifier::Classifier;

    using Encoder = ClassEncoder<OVA<Classifier>>;
    using Encoder::numClasses, Encoder::fit, Encoder::predict;


    void fit_ (const Mat& X, const Veci& y)
    {
        M = X.rows(), N = X.cols();

        classifiers.resize(numClasses, static_cast<Classifier&>(*this));

        Veci yk(M);


        for(int k = 0; k < numClasses; ++k)
        {
            std::transform(std::begin(y), std::end(y), std::begin(yk), [&](int x)
            {
                return x == k ? Classifier::classLabels[0] : Classifier::classLabels[1];
            });

            classifiers[k].fit_(X, yk);
        }
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

    static std::vector<int> classLabels;    
};

template <class Classifier>
std::vector<int> OVA<Classifier>::classLabels = {};


#endif // CPP_ML_OVA_H