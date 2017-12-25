#ifndef CPP_ML_BAYESIAN_LOGISTIC_REGRESSION_TWO_CLASS_H
#define CPP_ML_BAYESIAN_LOGISTIC_REGRESSION_TWO_CLASS_H

#include "../Classifier.h"

#include "../LogisticRegression/LogisticRegressionTwoClass.h"



namespace impl
{

template <bool Polymorphic = false>
struct BayesianLogisticRegressionTwoClass : public LogisticRegressionTwoClass<L2, Newton<>, Polymorphic>
{
    USING_LOGISTIC_REGRESSION(LogisticRegressionTwoClass<L2, Newton<>, Polymorphic>);

    using BaseLogisticRegression::w, BaseLogisticRegression::intercept, BaseLogisticRegression::predict, 
          BaseLogisticRegression::predictMargin, BaseLogisticRegression::optimizeFunc;


    void fit (const Mat& X, const Veci& y)
    {
        fit(X, y, 1e-4, 1e-4, 50);
    }


    void fit (const Mat& X, const Veci& y, double gTol, double aTol = 1e-4, int maxIter = 50)
    {
        optimize(X, y, gTol, aTol, maxIter);

        intercept = w(N-1);
        
        w.conservativeResize(N-1);

        snR = Sn.row(N-1).head(N-1);
        snC = Sn.col(N-1).head(N-1);
        snNN = Sn(N-1, N-1);

        Sn.conservativeResize(N-1, N-1);
    }



    void optimize (const Mat& X, const Veci& y, double gTol, double aTol, int maxIter)
    {
        w = Vec::Constant(N, 0.0);

        alpha = 0.0;

        double oldAlpha = alpha;

        Vec g;        

        auto func = optimizeFunc(X, y);

        
        for(int iter = 0; iter < maxIter; ++iter)
        {
            std::tie(g, Sn) = func(w);

            w = w - solveMat(Sn, g);
            

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




    // int predict (const Vec& x)
    // {
    //     return predictMargin(x) > 0.0;
    // }

    // Veci predict (const Mat& X)
    // {
    //     return (ArrayXd(predictMargin(X)) > 0.0).cast<int>();
    // }



    double predictProb (const Vec& x)
    {
        double dot = x.dot((Sn * x) + snR) + x.dot(snC) + snNN;

        return sigmoid(kappa(dot) * predictMargin(x));
    }

    Vec predictProb (const Mat& X)
    {
        return Vec::NullaryExpr(X.rows(), [&](int i){ return predictProb(Vec(X.row(i))); });
    }


    template <typename T>
    auto kappa (const T& sigma)
    {
        return sqrt(1.0 + pi() * (sigma / 8.0));
    }



    Mat Sn;

    Vec snR;
    Vec snC;

    double snNN;

};


}



template <bool EncodeLabels = true>
using BayesianLogisticRegressionTwoClass = impl::Classifier<impl::BayesianLogisticRegressionTwoClass<false>, EncodeLabels>;


namespace poly
{

template <bool EncodeLabels = true>
using BayesianLogisticRegressionTwoClass = impl::Classifier<impl::BayesianLogisticRegressionTwoClass<true>, EncodeLabels>;

}
              







#endif // CPP_ML_BAYESIAN_LOGISTIC_REGRESSION_TWO_CLASS_H