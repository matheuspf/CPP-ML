#ifndef CPP_ML_LOGISTIC_REGRESSION_MULTICLASS_H
#define CPP_ML_LOGISTIC_REGRESSION_MULTICLASS_H

#include "../../Modelo.h"

#include "../Classifier.h"

#include "LogisticRegressionBase.h"

#include <Eigen/Sparse>



namespace impl
{

template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>,
          bool EncodeLabels = true, bool Polymorphic = false>
struct LogisticRegressionMultiClass : public LogisticRegressionBase<Regularizer, Optimizer>,
                                      PickClassifierBase<LogisticRegressionMultiClass<Regularizer, Optimizer,
                                                         EncodeLabels, Polymorphic>, EncodeLabels, Polymorphic>
{
    USING_LOGISTIC_REGRESSION(LogisticRegressionBase<Regularizer, Optimizer>);
    USING_CLASSIFIER(PickClassifierBase<LogisticRegressionMultiClass<Regularizer, Optimizer,
                                        EncodeLabels, Polymorphic>, EncodeLabels, Polymorphic>);


    void fit_ (const Mat& X, const Veci& y)
    {
        M = X.rows(), N = X.cols();


        // auto func = [&](const Vec& w) -> double
        // {
        //     Mat W = Mat::Map(&w(0), numClasses, N);

        //     double res = 0;

        //     for(int i = 0; i < M; ++i)
        //         res += logSoftmax(W * X.row(i).transpose(), y(i));

        //     //db(-res, "     ", 0.5 * alpha * regularizer(w), "\n");

        //     return -res + 0.5 * alpha * regularizer(w);
        // };

        // auto grad = [&](const Vec& w) -> Vec
        // {
        //     Mat W = Mat::Map(&w(0), numClasses, N);
            
        //     Mat grad = Mat::Constant(numClasses, N, 0.0);

        //     for(int i = 0; i < M; ++i)
        //     {
        //         Vec sm = logSoftmax(W * X.row(i).transpose());

        //         sm(y(i)) -= 1.0;

        //         grad += sm * X.row(i);
        //     }
            
        //     return Vec::Map(&grad(0), numClasses * N) + 0.5 * alpha * regularizer.gradient(w);
        // };



        // Vec w0 = Vec::NullaryExpr(numClasses * N, [&](int){ return randDouble(-0.1, 0.1); });
        // //Vec w0 = Vec::Constant(numClasses * N, 0.0);

        // w0 = optimizer(func, grad, w0);
        

        // W = Mat::Map(&w0(0), numClasses, N);

        optimize(X, y);

        intercept = W.row(W.rows()-1);

        W.conservativeResize(W.rows()-1, Eigen::NoChange);
    }



    int predict_ (const Vec& x)
    {
        Vec vals = W.transpose() * x + intercept;

        double b = -1e20;
        int p = 0;

        for(int i = 0; i < numClasses; ++i)
            if(vals(i) > b)
                b = vals(i), p = i;

        return p;
    }

    Veci predict_ (const Mat& X)
    {
        Veci vals(X.rows());

        for(int i = 0; i < X.rows(); ++i)
            vals(i) = predict_(Vec(X.row(i)));

        return vals;
    }


    void optimize (const Mat& X, const Veci& y)
    {
        int K = numClasses;


        Eigen::SparseMatrix<double> T(M, K);

        std::vector<Eigen::Triplet<double>> vecTrip(M);

        int cnt = 0;

        std::generate(vecTrip.begin(), vecTrip.end(), [&]{ return Triplet<double>(cnt++, y(cnt), 1.0); });

        T.setFromTriplets(vecTrip.begin(), vecTrip.end());


        W = Mat::Constant(N, K, 0.0);

        Map<Vec> w(W.data(), W.size());


        Vec R = Vec::Constant(N, 0.0);

        Mat H = Mat::Constant(N*K, N*K, 0.0);

        Mat A, S, G;

        for(int i = 0; i < 50; ++i)
        {
            A = X * W;

            S = softmax(A);

            G = X.transpose() * (S - T) + alpha * W;

            Map<Vec> g(G.data(), G.size());

            for(int i = 0; i < K; ++i)
            {
                for(int j = 0; j < K; ++j)
                {
                    R = S.col(i).array() * ((i == j ? 1.0 : 0.0) - S.col(j).array());

                    H.block(i * N, j * N, N, N) = X.transpose() * R.asDiagonal() * X;
                }
            }

            H.diagonal().array() += alpha;
            //H = H + alpha * Mat::Identity(N*K, N*K);

            w = w - solveMat(H, g);

            if(g.norm() < 1e-8)
                break;
        }
    }


    Vec vec (const Mat& X)
    {
        Vec x(X.rows() * X.cols());

        for(int i = 0, k = 0; i < X.rows(); ++i)
            for(int j = 0; j < X.cols(); ++j, ++k)
                x(k) = X(i, j);

        return x;
    }

    Mat mat (const Vec& x, int r, int c)
    {
        Mat X(r, c);

        for(int i = 0, k = 0; i < r; ++i)
            for(int j = 0; j < c; ++j, ++k)
                X(i, j) = x(k);

        return X;
    }


    // double predictProb (const Vec& x)
    // {
    // }

    // Vec predictProb (const Mat& X)
    // {
    // }


    // double predictMargin (const Vec& x)
    // {
    // }

    // Vec predictMargin (const Mat& X)
    // {
    // }



    Mat W;

    Vec intercept;
};

} // namespace impl



template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>, bool EncodeLabels = true>
using LogisticRegressionMultiClass = impl::LogisticRegressionMultiClass<Regularizer, Optimizer, EncodeLabels, false>;


namespace poly
{
    template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>, bool EncodeLabels = true>
    using LogisticRegressionMultiClass = impl::LogisticRegressionMultiClass<Regularizer, Optimizer, EncodeLabels, true>;
}




#endif // CPP_ML_LOGISTIC_REGRESSION_MULTICLASS_H