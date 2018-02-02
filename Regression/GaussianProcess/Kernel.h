#pragma once

#include "../../Modelo.h"



struct LinearKernel
{
    
};


struct ExponentialKernel
{
    ExponentialKernel(const Vector4d phi = Vector4d{1.0, 4.0, 0.0, 5.0}) : phi(phi) {}


    void set (const Vec& x)
    {
        std::copy(x.data(), x.data() + x.size(), phi.data());
    }


    template <typename A, typename B>
    auto exec (const A& a, const B& b) const
    {
        using std::exp;
        using Eigen::exp;

        return phi(0) * exp(-phi(1) * a) + phi(2) + phi(3) * b;
        //return exp(-phi(1) * a);
    }


    double operator () (const Vec& x, const Vec& z) const
    {
        return exec((x - z).squaredNorm(), x.dot(z));
    }

    Vec operator () (const Mat& X, const Vec& x) const
    {
        return exec((X.rowwise() - x.transpose()).rowwise().squaredNorm().array(), (X * x).array());
    }

    Mat operator () (const Mat& X, const Mat& Z) const
    {
        Mat Y(X.rows(), Z.rows());

        for(int i = 0; i < X.rows(); ++i)
            Y.row(i) = (Z.rowwise() - X.row(i)).rowwise().squaredNorm();

        return exec(Y.array(), (X * Z.transpose()).array());
    }





    Vector4d phi;
};
