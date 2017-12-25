#ifndef CPP_ML_BAYESIAN_LOGISTIC_REGRESSION_MULTICLASS_H
#define CPP_ML_BAYESIAN_LOGISTIC_REGRESSION_MULTICLASS_H


#include "../LogisticRegression/LogisticRegressionMultiClass.h"



namespace impl
{

template <bool Polymorphic = false>
struct BayesianLogisticRegressionMultiClass : public LogisticRegressionMultiClass<L2, Newton<>, Polymorphic>
{
    USING_LOGISTIC_REGRESSION(LogisticRegressionMultiClass<L2, Newton<>, Polymorphic>);

    using BaseLogisticRegression::W, BaseLogisticRegression::intercept, BaseLogisticRegression::predict, 
          BaseLogisticRegression::predictMargin;

    using OptimizeFunc = typename BaseLogisticRegression::OptimizeFunc;


    void fit (const Mat& X, const Veci& y)
    {
        fit(X, y, 1e-6, 1e-4, 50);
    }


    void fit (const Mat& X, const Veci& y, double gTol, double aTol = 1e-4, int maxIter = 50)
    {
        M = X.rows(), N = X.cols();

        optimize(X, y, gTol, aTol, maxIter);

        intercept = W.row(W.rows()-1);

        W.conservativeResize(W.rows()-1, Eigen::NoChange);
    }



    void optimize (const Mat& X, const Veci& y, double gTol, double aTol, int maxIter)
    {
        W = Mat::Constant(N, numClasses, 0.0);

        Map<Vec> w(W.data(), W.size());

        alpha = 0.0;

        double oldAlpha = alpha;

        OptimizeFunc func(N, M, numClasses, alpha, X, y);

        
        for(int iter = 0; iter < maxIter; ++iter)
        {
            const auto& [G, H] = func(W);

            Map<Vec> g(G.data(), G.size());

            w = w - solveMat(H, g);
            

            Sn = H;

            Sn.diagonal().array() -= alpha;

            Eigen::EigenSolver<Mat> eigSolver(Sn);

            const ArrayXd& eigVals = eigSolver.eigenvalues().real().array();

            double gamma = (eigVals / (alpha + eigVals)).sum();

            oldAlpha = alpha;

            alpha = gamma / w.dot(w);


            if(g.norm() < gTol && std::abs(alpha - oldAlpha) < aTol)
                break;
        }
    }




    // double predictProb (const Vec& x)
    // {
    //     return sigmoid(kappa(x.dot(Sn * x)) * predictMargin(x));
    // }

    // Vec predictProb (const Mat& X)
    // {
    //     return Vec::NullaryExpr(X.rows(), [&](int i){ return predictProb(Vec(X.row(i))); });
    // }




    // int predict (const Vec& x)
    // {
    //     Vec vals = predictMargin(x);
        
    //     return std::max_element(std::begin(vals), std::end(vals)) - std::begin(vals);
    // }

    // Veci predict (const Mat& X)
    // {
    //     Mat vals = predictMargin(X);
    //     vals.transposeInPlace();

    //     return Veci::NullaryExpr(X.rows(), [&](int i)
    //     {
    //         return std::max_element(&vals.col(i)(0), &vals.col(i)(0) + numClasses) - &vals.col(i)(0);
    //     });
    // }





    template <typename T>
    auto kappa (const T& sigma)
    {
        return sqrt(1.0 + pi() * (sigma / 8.0));
    }



    Mat Sn;

};


}



template <bool EncodeLabels = true>
using BayesianLogisticRegressionMultiClass = impl::Classifier<impl::BayesianLogisticRegressionMultiClass<false>, EncodeLabels>;


namespace poly
{

template <bool EncodeLabels = true>
using BayesianLogisticRegressionMultiClass = impl::Classifier<impl::BayesianLogisticRegressionMultiClass<true>, EncodeLabels>;
    
}







#endif // CPP_ML_BAYESIAN_LOGISTIC_REGRESSION_MULTICLASS_H