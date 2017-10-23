#ifndef CPP_ML_OVO_H
#define CPP_ML_OVO_H

#include "../../Modelo.h"

#include "../../Preprocessing/Preprocess.h"


template <class Classifier>
struct OVO : public Classifier
{
public:

    using Classifier::Classifier, Classifier::numClasses;


    void fit_ (const Mat& X, const Veci& y)
    {
        M = X.rows(), N = X.cols();

        classifiers = MatX<Classifier>::Constant(numClasses, numClasses, static_cast<Classifier&>(*this));

        std::vector<std::vector<int>> indexes(numClasses);

        for(int i = 0; i < M; ++i)
            indexes[y(i)].push_back(i);

        
        std::vector<int> ids;
        ids.reserve(M);

        for(int k = 0; k < numClasses; ++k)
        {
            for(int l = k+1; l < numClasses; ++l)
            {
                ids.clear();
                ids.insert(ids.end(), indexes[k].begin(), indexes[k].end());
                ids.insert(ids.end(), indexes[l].begin(), indexes[l].end());

                Mat Xk = index(X, ids);
                Veci yk = index(y, ids);

                std::transform(std::begin(yk), std::end(yk), std::begin(yk), [k](int x){ return x == k ? 1 : 0; });

                classifiers(k, l).fit(Xk, yk);
            }
        }
    }



    // Veci predict_ (const Mat& X)
    // {
    //     MatX<VecX<pair<int, double>>> classification(numClasses, numClasses);
        
    //     for(int k = 0; k < numClasses; ++k)
    //         for(int l = k+1; l < numClasses; ++l)
    //         {
    //             Veci lb = classifiers(k, l).predict(X);
    //             Vec pb = classifiers(k, l).predictProb(X);

    //             classification(k, l) = VecX<pair<int, double>>::Constant(X.rows(), pair<int, double>(0, 0.0));

    //             for(int i = 0; i < X.rows(); ++i)
    //             {
    //                 classification(k, l)(i).first = lb(i);
    //                 classification(k, l)(i).second = pb(i);
    //             }
    //         }
        
    //     MatX<pair<int, double>> probs = MatX<pair<int, double>>::Constant(numClasses, X.rows(), pair<int, double>(0, 0.0));

    //     for(int k = 0; k < numClasses; ++k)
    //         for(int l = k+1; l < numClasses; ++l)
    //             for(int i = 0; i < X.rows(); ++i)
    //             {
    //                 (classification(k, l)(i).first ? probs(k, i).first : probs(l, i).first)++;

    //                 probs(k, i).second += classification(k, l)(i).second;
    //                 probs(l, i).second += 1.0 - classification(k, l)(i).second;
    //             }
        

    //     //probs.transposeInPlace();

    //     FOR(i, numClasses) FORR(j, i+1, numClasses)
    //         swap(probs(i, j), probs(j, i));


    //     Veci labels(X.rows());

    //     for(int i = 0; i < X.rows(); ++i)
    //         labels(i) = std::max_element(&probs(i, 0), &probs(i, numClasses-1)+1) - &probs(i, 0);


    //     return labels;
    // }


    Veci predict_ (const Mat& X)
    {db("WOW");
        MatX<Vec> probs(numClasses, numClasses);
        Mat vals = Mat::Constant(X.rows(), numClasses, 0.0);

        for(int k = 0; k < numClasses; ++k)
        {
            for(int l = k+1; l < numClasses; ++l)
            {
                probs(k, l) = Veci(classifiers(k, l).predict(X)).cast<double>();
                probs(l, k) = (1.0 - probs(k, l).array());

                probs(k, l) *= 10.0;
                probs(l, k) *= 10.0;

                probs(k, l) = probs(k, l) + classifiers(k, l).predictProb(X);
                probs(l, k) = probs(l, k).array() + (1.0 - classifiers(k, l).predictProb(X).array());
            }

            for(int i = 0; i < X.rows(); ++i)
                for(int l = 0; l < numClasses; ++l) if(k != l)
                    vals(i, k) += probs(k, l)(i);
        }
        
        Veci labels(X.rows());

        for(int i = 0; i < X.rows(); ++i)
        {
            double val = -1e20;

            for(int k = 0; k < numClasses; ++k)
                if(vals(i, k) > val)
                    val = vals(i, k), labels(i) = k;
        }

        return labels;
    }




    int predict_ (const Vec&) { return 0; }


    Vec predictMargin (const Mat&) { return Vec(); }
    Vec predictProb (const Mat&) { return Vec(); }



    int M, N;

    MatX<Classifier> classifiers;
};



#endif // CPP_ML_OVO_H