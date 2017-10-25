#ifndef CPP_ML_OVO_H
#define CPP_ML_OVO_H

#include "../../Modelo.h"

#include "../../Preprocessing/Preprocess.h"

#include "../Classifier.h"



template <class Cls>
struct OVO : public std::conditional_t<std::is_polymorphic<Cls>::value, Cls, Classifier<OVA<Cls>>>
{
public:

    using Base = std::conditional_t<std::is_polymorphic<Cls>::value, Cls, Classifier<OVA<Cls>>>;
    using Base::numClasses, Base::fit, Base::predict, Base::positiveClass, Base::negativeClass;


    template <typename... Args>
    OVO (Args&&... args) : baseClassifier(std::forward<Args>(args)...) {}


    void fit_ (const Mat& X, const Veci& y)
    {
        M = X.rows(), N = X.cols();

        classifiers = MatX<Cls>::Constant(numClasses, numClasses, baseClassifier);

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

                std::transform(std::begin(yk), std::end(yk), std::begin(yk), [&](int x)
                {
                    return x == k ? positiveClass : negativeClass;
                });

                classifiers(k, l).fit_(Xk, yk);
            }
        }
    }



    Veci predict_ (const Mat& X)
    {
        MatX<Vec> probs(numClasses, numClasses);
        Mat vals = Mat::Constant(X.rows(), numClasses, 0.0);

        for(int k = 0; k < numClasses; ++k)
        {
            for(int l = k+1; l < numClasses; ++l)
            {
                probs(k, l) = classifiers(k, l).predictProb(X);
                probs(l, k) = (1.0 - probs(k, l).array());
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



    Cls baseClassifier;    

    int M, N;

    MatX<Cls> classifiers;
};



#endif // CPP_ML_OVO_H