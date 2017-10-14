#ifndef CPP_ML_REGULARIZERS_H
#define CPP_ML_REGULARIZERS_H

#include "Modelo.h"



struct L1
{
    template <class T, int Rows, int Cols>
    double operator () (const Eigen::Matrix<T, Rows, Cols>& X)
    {
        return X.array().abs().sum();
    }

    template <class T, int Rows, int Cols>
    double cost (const Eigen::Matrix<T, Rows, Cols>& X)
    {
        return operator()(X);
    }

    template <class T, int Rows, int Cols>
    Eigen::Matrix<T, Rows, Cols> gradient (const Eigen::Matrix<T, Rows, Cols>& X)
    {
        return select(X, [](const auto& x){ return T((x >= 0.0 ? 1.0 : -1.0)); });
    }
};


struct L2
{
    template <class T, int Rows, int Cols>
    double operator () (const Eigen::Matrix<T, Rows, Cols>& X)
    {
        return X.squaredNorm();
    }

    template <class T, int Rows, int Cols>
    double cost (const Eigen::Matrix<T, Rows, Cols>& X)
    {
        return operator()(X);
    }

    template <class T, int Rows, int Cols>
    Eigen::Matrix<T, Rows, Cols> gradient (const Eigen::Matrix<T, Rows, Cols>& X)
    {
        return 2 * X;
    }
};


struct LInf
{
    template <class T, int Rows, int Cols>
    double operator () (const Eigen::Matrix<T, Rows, Cols>& X)
    {
        return X.array().abs().maxCoeff();
    }

    template <class T, int Rows, int Cols>
    double cost (const Eigen::Matrix<T, Rows, Cols>& X)
    {
        return operator()(X);
    }
};



#endif // CPP_ML_REGULARIZERS_H