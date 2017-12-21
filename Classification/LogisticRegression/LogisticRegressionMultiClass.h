#ifndef CPP_ML_LOGISTIC_REGRESSION_MULTICLASS_H
#define CPP_ML_LOGISTIC_REGRESSION_MULTICLASS_H


#include "LogisticRegressionBase.h"

#include <Eigen/Sparse>



namespace impl
{

template <class Regularizer = L2, class Optimizer = Newton<>, bool Polymorphic = true>
struct LogisticRegressionMultiClass : public LogisticRegressionBase<Regularizer, Optimizer, Polymorphic>
{
    USING_LOGISTIC_REGRESSION(LogisticRegressionBase<Regularizer, Optimizer, Polymorphic>);

    
    void fit (const Mat& X, const Veci& y)
    {
        fit(X, y, 1e-6, 50);
    }

    void fit (const Mat& X, const Veci& y, double gTol, int maxIter = 50)
    {
        M = X.rows(), N = X.cols();

        optimize(X, y, gTol, maxIter);

        intercept = W.row(W.rows()-1);

        W.conservativeResize(W.rows()-1, Eigen::NoChange);
    }



    int predict (const Vec& x)
    {
        Vec vals = predictMargin(x);
        
        return std::max_element(std::begin(vals), std::end(vals)) - std::begin(vals);
    }

    Veci predict (const Mat& X)
    {
        Mat vals = predictMargin(X);
        vals.transposeInPlace();

        return Veci::NullaryExpr(X.rows(), [&](int i)
        {
            return std::max_element(&vals.col(i)(0), &vals.col(i)(0) + numClasses) - &vals.col(i)(0);
        });
    }



    void optimize (const Mat& X, const Veci& y, double gTol, int maxIter)
    {
        W = Mat::Constant(N, numClasses, 0.0);

        Map<Vec> w(W.data(), W.size());

        OptimizeFunc func(N, M, numClasses, alpha, X, y);


        for(int iter = 0; iter < maxIter; ++iter)
        {
            const auto& [G, H] = func(W);

            Map<Vec> g(G.data(), G.size());

            w = w - solveMat(H, g);

            if(g.norm() < gTol)
                break;
        }
    }



    Vec predictProb (const Vec& x)
    {
        return softmax(predictMargin(x));
    }

    Mat predictProb (const Mat& X)
    {
        return softmax(predictMargin(X));        
    }


    Vec predictMargin (const Vec& x)
    {
        return W.transpose() * x + intercept;
    }

    Mat predictMargin (const Mat& X)
    {
        return (X * W).rowwise() + intercept.transpose();
    }



    struct OptimizeFunc
    {
        OptimizeFunc (int N, int M, int K, const double& alpha, const Mat& X, const Veci& y) : 
                      N(N), M(M), K(K), alpha(alpha), X(X), y(y), R(Vec::Constant(N, 0.0)), 
                      H(Mat::Constant(N*K, N*K, 0.0)), T(M, K), vecTrip(M)
        {
            int cnt = 0;

            std::generate(vecTrip.begin(), vecTrip.end(), [&]{ return Triplet<double>(cnt++, y(cnt), 1.0); });

            T.setFromTriplets(vecTrip.begin(), vecTrip.end());
        }


        inline auto operator () (const Mat& W)
        {
            A = X * W;

            S = softmax(A);

            G = X.transpose() * (S - T) + alpha * W;

            for(int i = 0; i < K; ++i)
            {
                for(int j = 0; j < K; ++j)
                {
                    R = S.col(i).array() * ((i == j ? 1.0 : 0.0) - S.col(j).array());

                    H.block(i * N, j * N, N, N) = X.transpose() * R.asDiagonal() * X;
                }
            }

            H.diagonal().array() += alpha;

            return std::tie(G, H);
        }



        int N, M, K;

        const double& alpha;        


        const Mat& X;

        const Veci& y;


        Vec R;

        Mat A, S, G, H;

        Eigen::SparseMatrix<double> T;

        std::vector<Eigen::Triplet<double>> vecTrip;
    };






    Mat W;

    Vec intercept;
};

} // namespace impl



template <class Regularizer = L2, class Optimizer = Newton<>, bool EncodeLabels = true>
using LogisticRegressionMultiClass = impl::Classifier<impl::LogisticRegressionMultiClass<Regularizer, Optimizer, false>, 
                                                      EncodeLabels>;


namespace poly
{
template <class Regularizer = L2, class Optimizer = Newton<>, bool EncodeLabels = true>
using LogisticRegressionMultiClass = impl::Classifier<impl::LogisticRegressionMultiClass<Regularizer, Optimizer, true>, 
                                                      EncodeLabels>;
}



#endif // CPP_ML_LOGISTIC_REGRESSION_MULTICLASS_H