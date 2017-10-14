#ifndef CPP_ML_OVO_H
#define CPP_ML_OVO_H

#include "../../Modelo.h"

#include "../../Preprocessing/Preprocess.h"


struct OVO
{

    OVO& fit (const Mat& X, const Veci& y)
    {
        lenc.fit(y_);
        
        K = lenc.K;
        classifiers.resize(K*(K-1));

        Mat Xk(X_.rows(), X_.cols());
        Veci yk(y_i.rows());


        for(int i = 0; i < K; ++i)
        {
            for(int j = i+1; j < K; ++j)
            {
                
            }
        }
    }


    int K;
    
    LabelEncoder<int> lenc;

    std::vector<Classifier> classifiers;

    std::vector<double> classProb;
};



#endif // CPP_ML_OVO_H