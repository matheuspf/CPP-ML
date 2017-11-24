#ifndef CPP_ML_UNSUPERVISED_FISHERS_LDA
#define CPP_ML_UNSUPERVISED_FISHERS_LDA

#include "../../Modelo.h"
#include "../../Preprocessing/Preprocess.h"
#include "../../SpectraHelpers.h"


struct FishersLDAMultiClass
{
    FishersLDAMultiClass (int D = -1) : D(D) {}


    FishersLDAMultiClass& fit (const Mat& X, Veci y)
    {
        y = lenc.fitTransform(y);
        
        int M = X.rows(), N = X.cols(), K = lenc.numClasses;

        std::vector<Vec> means(K, Vec::Constant(N, 0.0));
        std::vector<int> Ns(K);

        Mat Sw = Mat::Constant(N, N, 0.0);
        Mat Sb = Mat::Constant(N, N, 0.0);

        Vec mean = Vec::Constant(N, 0.0);
        Vec diff(N);


        for(int i = 0; i < M; ++i)
        {
            means[y(i)] += X.row(i).transpose();
            Ns[y(i)]++;
        }

        for(int i = 0; i < K; ++i)
            mean += means[i];

        for(int i = 0; i < K; ++i)
        {
            diff = means[i] - mean;
            Sb += Ns[i] * diff * diff.transpose();
        }

        for(int i = 0; i < M; ++i)
        {
            diff = X.row(i).transpose() - means[y(i)];
            Sw += diff * diff.transpose();
        }


        TopEigen<> topEigen(inverseMat(Sw) * Sb, D <= 1 ? K-1 : std::min(D, K-1));

        W = topEigen.eigenvectors();


        return *this;
    }


    Vec transform (const Vec& x)
    {
        return x * W;
    }

    Mat transform (const Mat& X)
    {
        return X * W;
    }



    auto fitTransform (const Mat& X, const Veci& y)
    {
        return fit(X, y).transform(X);
    }




    int D;

    LabelEncoder<int> lenc;

    Mat W;
};




#endif //CPP_ML_UNSUPERVISED_FISHERS_LDA