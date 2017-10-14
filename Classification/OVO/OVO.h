#ifndef CPP_ML_OVO_H
#define CPP_ML_OVO_H

#include "../../Modelo.h"

#include "../../Preprocessing/Preprocess.h"


template <class Classifier>
struct OVO
{
public:

    template <typename... Args>
    OVO (Args&&... args) : classifiers(1, Classifier(std::forward<Args>(args)...)) {}


    template <typename... Args>
    OVO& fit (const Mat& X, Veci y, Args&&... args)
    {
        M = X.rows(), N = X.cols();

        y = lenc.fitTransform(y);

        K = lenc.K;
        classifiers.resize(K*(K-1), classifiers[0]);


        std::vector<std::vector<int>> indexes(K);

        for(int i = 0; i < M; ++i)
            indexes[y(i)].push_back(i);

        
        std::vector<int> ids;
        ids.reserve(M);

        for(int k = 0, p = 0; k < K; ++k)
        {
            for(int l = k+1; l < K; ++l, ++p)
            {
                ids.clear();
                ids.insert(ids.end(), indexes[k].begin(), indexes[k].end());
                ids.insert(ids.end(), indexes[l].begin(), indexes[l].end());

                Mat Xkl = index(X, ids);

                classifiers[p].fit(X, yk, std::forward<Args>(args)...);
            }
        }

        return *this;
    }


    template <typename... Args>
    Veci predict (const Mat& X, Args&&... args)
    {
        Mat classProb(X.rows(), K*(K-1));

        for(int k = 0; k < classifiers.size(); ++k)
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



#endif // CPP_ML_OVO_H