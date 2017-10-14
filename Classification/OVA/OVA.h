#ifndef CPP_ML_OVA_H
#define CPP_ML_OVA_H

#include "../../Modelo.h"

#include "../../Preprocessing/Preprocess.h"


template <class Classifier>
struct OVA
{
public:

    template <typename... Args>
    OVA (Args&&... args) : classifiers(1, Classifier(std::forward<Args>(args)...)) {}


    template <typename... Args>
    OVA& fit (const Mat& X, Veci y, Args&&... args)
    {
        M = X.rows(), N = X.cols();

        y = lenc.fitTransform(y);

        K = lenc.K;
        classifiers.resize(K, classifiers[0]);


        Veci yk(M);


        for(int k = 0; k < K; ++k)
        {
            std::transform(std::begin(y), std::end(y), std::begin(yk), [&](int x)
            {
                return x == k ? 1 : 0;
            });

            classifiers[k].fit(X, yk, std::forward<Args>(args)...);
        }
    }


    template <typename... Args>
    Veci predict (const Mat& X, Args&&... args)
    {
        Mat classProb(X.rows(), K);

        for(int k = 0; k < K; ++k)
            classProb.col(k) = classifiers[k].predictMargin(X, std::forward<Args>(args)...);

        Veci labels(X.rows());

        for(int i = 0; i < X.rows(); ++i)
        {
            double val = -1e20;
            
            for(int k = 0; k < K; ++k)
            {
                if(classProb(i, k) > val)
                {
                    val = classProb(i, k);
                    labels(i) = k;
                }
            }

            labels(i) = lenc.reverseMap[labels(i)];
        }

        return labels;
    }




    int M, N, K;

    LabelEncoder<int> lenc;

    std::vector<Classifier> classifiers;
};



#endif // CPP_ML_OVA_H