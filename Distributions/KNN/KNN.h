#ifndef CPP_ML_KNN_H
#define CPP_ML_KNN_H

#include "../../Modelo.h"

#include "../../nanoflann.hpp"




struct KNN
{
    KNN (int K = 3) : K(K) {}

    KNN (const KNN& kd) : X(kd.X), K(kd.K), 
    kdTree(kd.kdTree ? std::make_unique<nanoflann::KDTreeEigenMatrixAdaptor<Mat>>(kd.X, 10) : nullptr) {}


    KNN& operator = (const KNN& kd)
    {
        X = kd.X;
        K = kd.K;
        kdTree = kd.kdTree ? std::make_unique<nanoflann::KDTreeEigenMatrixAdaptor<Mat>>(kd.X, 10) : nullptr;
    }


    void fit (const Mat& X_)
    {
        X = X_;

        update();
    }


    auto params ()
    {
        return std::make_tuple(X, K);
    }

    void params (const Mat& X_, int K_ = 1.0)
    {
        X = X_;
        K = K_;

        update();
    }

    void update ()
    {
        kdTree = std::make_unique<nanoflann::KDTreeEigenMatrixAdaptor<Mat>>(X, 10);
        kdTree->index->buildIndex();
    }


    double operator () (const Vec& x)
    {
        const auto& [indices, dists] = getIndices(x);
        
        return K / std::max(*std::max_element(dists.begin(), dists.end()), 1e-8);

        // return std::accumulate(indices.begin(), indices.end(), 0.0, [&](double sum, const auto& p)
        // {
        //     return sum + kernel(x, X.row(p.first));
        // });
    }

    Vec operator () (const Mat& X)
    {
        return Vec::NullaryExpr(X.rows(), [&](int i){ return operator()(Vec(X.row(i))); });
    }



    std::pair<const std::vector<Eigen::Index>&, const std::vector<double>&> getIndices (const Vec& x)
    {
        static std::vector<Eigen::Index> indices(K);
        static std::vector<double> dists(K);

        kdTree->index->knnSearch(&x(0), K, &indices[0], &dists[0]);


        return std::pair<const std::vector<Eigen::Index>&, const std::vector<double>&>(indices, dists);
    }



    Mat X;

    int K;

    std::unique_ptr<nanoflann::KDTreeEigenMatrixAdaptor<Mat>> kdTree;
};



#endif // CPP_ML_KNN_H