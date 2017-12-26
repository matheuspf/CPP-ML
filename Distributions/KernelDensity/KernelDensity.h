#ifndef CPP_ML_KERNEL_DENSITY_H
#define CPP_ML_KERNEL_DENSITY_H

#include "../../Modelo.h"

#include "../../nanoflann.hpp"




struct KernelDensity
{
    KernelDensity (double sigma = 1.0) : sigma(sigma) {}

    KernelDensity (const KernelDensity& kd) : X(kd.X), sigma(kd.sigma), 
    kdTree(kd.kdTree ? std::make_unique<nanoflann::KDTreeEigenMatrixAdaptor<Mat>>(kd.X, 10) : nullptr) {}


    KernelDensity& operator = (const KernelDensity& kd)
    {
        X = kd.X;
        sigma = kd.sigma;
        kdTree = kd.kdTree ? std::make_unique<nanoflann::KDTreeEigenMatrixAdaptor<Mat>>(kd.X, 10) : nullptr;
    }


    void fit (const Mat& X_)
    {
        X = X_;

        update();
    }


    auto params ()
    {
        return std::make_tuple(X, sigma);
    }

    void params (const Mat& X_, double sigma_ = 1.0)
    {
        X = X_;
        sigma = sigma_;

        update();
    }

    void update ()
    {
        kdTree = std::make_unique<nanoflann::KDTreeEigenMatrixAdaptor<Mat>>(X, 10);
        kdTree->index->buildIndex();
    }


    double operator () (const Vec& x)
    {
        const auto& indices = getIndices(x);

        return std::accumulate(indices.begin(), indices.end(), 0.0, [&](double sum, const auto& p)
        {
            return sum + kernel(x, X.row(p.first));
        });
    }

    Vec operator () (const Mat& X)
    {
        return Vec::NullaryExpr(X.rows(), [&](int i){ return operator()(Vec(X.row(i))); });
    }



    const std::vector<std::pair<Eigen::Index, double>>& getIndices (const Vec& x)
    {
        static std::vector<std::pair<Eigen::Index, double>> indicesDists;

        kdTree->index->radiusSearch(&x(0), 3 * std::sqrt(sigma), indicesDists, nanoflann::SearchParams());

        return indicesDists;
    }


    double kernel (const Vec& x1, const Vec& x2)
    {
        return (1.0 / std::sqrt(2 * pi() * sigma)) * std::exp(-(x1 - x2).squaredNorm() / (2 * sigma));
    }



    Mat X;

    double sigma;

    std::unique_ptr<nanoflann::KDTreeEigenMatrixAdaptor<Mat>> kdTree;
};



#endif // CPP_ML_KERNEL_DENSITY_H