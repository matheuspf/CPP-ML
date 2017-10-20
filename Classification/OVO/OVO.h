#ifndef CPP_ML_OVO_H
#define CPP_ML_OVO_H

#include "../../Modelo.h"

#include "../../Preprocessing/Preprocess.h"


template <class Classifier>
struct OVO
{
public:

    template <typename... Args>
    OVO (Args&&... args) : classifierBase(std::forward<Args>(args)...) {}


    template <typename... Args>
    OVO& fit (const Mat& X, Veci y, int K_ = 0, Args&&... args)
    {
        M = X.rows(), N = X.cols();

        K = K_;
        preProcessLabels = K <= 0;

        if(preProcessLabels)
        {
            y = lenc.fitTransform(y);
            K = lenc.K;
        }

        classifiers = MatX<Classifier>::Constant(K, K, classifierBase);

        std::vector<std::vector<int>> indexes(K);

        for(int i = 0; i < M; ++i)
            indexes[y(i)].push_back(i);

        
        std::vector<int> ids;
        ids.reserve(M);

        for(int k = 0; k < K; ++k)
        {
            for(int l = k+1; l < K; ++l)
            {
                ids.clear();
                ids.insert(ids.end(), indexes[k].begin(), indexes[k].end());
                ids.insert(ids.end(), indexes[l].begin(), indexes[l].end());

                Mat Xk = index(X, ids);
                Veci yk = index(y, ids);

                std::transform(std::begin(yk), std::end(yk), std::begin(yk), [k](int x){ return x == k ? 1 : 0; });

                classifiers(k, l).fit(Xk, yk, std::forward<Args>(args)...);
            }
        }

        return *this;
    }


    template <typename... Args>
    Veci predict (const Mat& X, Args&&... args)
    {
        MatX<VecX<pair<int, double>>> classification(K, K);
        
        for(int k = 0; k < K; ++k)
            for(int l = k+1; l < K; ++l)
            {
                Veci lb = classifiers(k, l).predict(X);
                Vec pb = classifiers(k, l).predictProb(X);

                classification(k, l) = VecX<pair<int, double>>(X.rows());

                for(int i = 0; i < X.rows(); ++i)
                {
                    classification(k, l)(i).first = lb(i);
                    classification(k, l)(i).second = pb(i);
                }
            }
        
        MatX<pair<int, double>> probs = MatX<pair<int, double>>::Constant(K, X.rows(), pair<int, double>(0, 0.0));

        for(int k = 0; k < K; ++k)
            for(int l = k+1; l < K; ++l)
                for(int i = 0; i < X.rows(); ++i)
                {
                    (classification(k, l)(i).first ? probs(k, i).first : probs(l, i).first)++;

                    probs(k, i).second += classification(k, l)(i).second;
                    probs(l, i).second += 1.0 - classification(k, l)(i).second;
                }
        

        probs.transposeInPlace();

        Veci labels(X.rows());


        for(int i = 0; i < X.rows(); ++i)
            labels(i) = std::max_element(&probs(i, 0), &probs(i, K-1)+1) - &probs(i, 0);

        if(preProcessLabels)
            std::transform(std::begin(labels), std::end(labels), std::begin(labels), [&](int x){ return lenc.reverseMap[x]; });

        return labels;
    }

    int predict (const Vec&) { return 0; }


    Vec predictMargin (const Mat&) { return Vec(); }
    Vec predictProb (const Mat&) { return Vec(); }



    int M, N, K;

    LabelEncoder<int> lenc;

    MatX<Classifier> classifiers;

    Classifier classifierBase;

    bool preProcessLabels;
};



#endif // CPP_ML_OVO_H