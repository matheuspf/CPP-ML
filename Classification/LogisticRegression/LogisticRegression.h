#ifndef CPP_ML_LOGISTIC_REGRESSION_H
#define CPP_ML_LOGISTIC_REGRESSION_H

#include "../../Modelo.h"

#include "../../Preprocessing/Preprocess.h"

#include "../../Optimization/Newton/Newton.h"

#include "../../Regularizers.h"

#include "../OVA/OVA.h"



template <class, class>
struct LogisticRegressionBase;

template <class, class>
struct LogisticRegressionTwoClass;

template <template <class> class, class>
struct LogisticRegressionHandler;



template <class Regularizer = L2, class Optimizer = Newton<Goldstein, CholeskyIdentity>>
struct LogisticRegression
{
    LogisticRegression (double alpha = 1e-8, const Optimizer& optimizer = Optimizer(), std::string multiClassType = "OVA") :
                        alpha(alpha), optimizer(optimizer), multiClassType(multiClassType) {}

    LogisticRegression (const Optimizer& optimizer, double alpha = 1e-8, std::string multiClassType = "OVA") :
                        alpha(alpha), optimizer(optimizer), multiClassType(multiClassType) {}

    LogisticRegression (std::string multiClassType = "OVA", double alpha = 1e-8, const Optimizer& optimizer = Optimizer()) :
                        alpha(alpha), optimizer(optimizer), multiClassType(multiClassType) {}


    void fit (Mat X, Veci y, int K_ = 0)
    {
        assert(X.rows() == y.rows() && "Observation matrix and labels differ in number of samples.");
        
        X.conservativeResize(Eigen::NoChange, X.cols()+1);
        X.col(X.cols()-1).array() = 1.0;

        K = K_;
        preProcessLabels = K <= 0;

        if(preProcessLabels)
        {
            y = lenc.fitTransform(y);
            K = lenc.K;
        }


        if(K == 2)
            impl = new LogisticRegressionTwoClass<Regularizer, Optimizer>(alpha, optimizer);
        
        else
        {
            if(multiClassType == "OVA")
                impl = new LogisticRegressionHandler<OVA, LogisticRegressionTwoClass<Regularizer, Optimizer>>(alpha, optimizer);
        }
        

        impl->fit(X, y, K);
    }



    int predict (const Vec& x)
    {
        int label = impl->predict(x);

        if(preProcessLabels)
            label = lenc.reverseMap[label];
        
        return label;
    }

    Veci predict (const Mat& X)
    {
        Veci labels = impl->predict(X);

        if(preProcessLabels)
        {
            std::transform(std::begin(labels), std::end(labels), std::begin(labels), [&](int label)
            {
                return lenc.reverseMap[label];
            });
        }

        return labels;
    }



    // double predictProb (const Vec& x)
    // {
    //     return impl->predictProb(x);
    // }

    // Vec predictProb (const Mat& X)
    // {
    //     return impl->predictProb(X);
    // }


    // double predictMargin (const Vec& x)
    // {
    //     return return impl->predictMargin(x);
    // }

    // Vec predictMargin (const Mat& X)
    // {
    //     return impl->predictMargin(X);
    // }
    

    
    //std::unique_ptr<LogisticRegressionBase<Regularizer, Optimizer>> impl;
    LogisticRegressionBase<Regularizer, Optimizer>* impl;

    double alpha;

    Regularizer regularizer;

    Optimizer optimizer;

    std::string multiClassType;
    
    LabelEncoder<int> lenc;

    int K;

    bool preProcessLabels;
};




template <class Regularizer, class Optimizer>
struct LogisticRegressionBase
{
    LogisticRegressionBase (double alpha = 1e-8, const Optimizer& optimizer = Optimizer()) :
                                alpha(alpha), optimizer(optimizer) {}

    LogisticRegressionBase (const Optimizer& optimizer, double alpha = 1e-8) :
                                alpha(alpha), optimizer(optimizer) {}


    virtual ~LogisticRegressionBase() {}

    LogisticRegressionBase (const LogisticRegressionBase&) = default;
    LogisticRegressionBase (LogisticRegressionBase&&)      = default;

    LogisticRegressionBase& operator= (const LogisticRegressionBase&) = default;
    LogisticRegressionBase& operator= (LogisticRegressionBase&&)      = default;

    
    virtual void fit (const Mat&, const Veci&, int = 0) = 0;

    virtual int predict (const Vec&)  = 0;
    virtual Veci predict (const Mat&) = 0;

    // virtual double predictProb (const Vec&) = 0;
    // virtual Vec predictProb (const Mat&)    = 0;
    
    // virtual double predictMargin (const Vec&) = 0;    
    // virtual Vec predictMargin (const Vec&)    = 0; 


    template <class T>
    auto sigmoid (const T& x)
    {
        using std::exp;
        using Eigen::exp;

        return 1.0 / (1.0 + exp(-x));
    }



    int M, N;

    double alpha;

    Regularizer regularizer;

    Optimizer optimizer;
};




template <class Regularizer, class Optimizer>
struct LogisticRegressionTwoClass : public LogisticRegressionBase<Regularizer, Optimizer>
{
    using Base = LogisticRegressionBase<Regularizer, Optimizer>;
    using Base::Base, Base::M, Base::N, Base::alpha, Base::regularizer, Base::optimizer, Base::sigmoid;



    void fit (const Mat& X, const Veci& y, int = 0)
    {
        M = X.rows(), N = X.cols();


        auto func = [&](const Vec& w) -> double
        {
            ArrayXd sig = sigmoid((X * w).array());

            return -(y.array().cast<double>() * log(sig) + (1.0 - y.array().cast<double>()) * log(1.0 - sig)).sum()
                    + 0.5 * alpha * regularizer(w);
                    //+ 0.5 * alpha * regularizer(Vec(w.head(N-1)));
        };

        auto grad = [&](const Vec& w) -> Vec
        {
            return X.transpose() * (sigmoid((X * w).array()).matrix() - y.cast<double>()) +
                    0.5 * alpha * regularizer.gradient(w);
        };


        w = Vec::Constant(N, 0.0);
        
        w = optimizer(func, grad, w);


        intercept = w(N-1);

        w.conservativeResize(N-1);
    }



    int predict (const Vec& x)
    {
        return predictMargin(x) > 0.0;
    }

    Veci predict (const Mat& X)
    {
        return (ArrayXd(predictMargin(X)) > 0.0).cast<int>();
    }



    double predictProb (const Vec& x)
    {
        return sigmoid(predictMargin(x));
    }

    Vec predictProb (const Mat& X)
    {
        return sigmoid(predictMargin(X));
    }


    double predictMargin (const Vec& x)
    {
        return w.dot(x) + intercept;
    }

    Vec predictMargin (const Mat& X)
    {
        return (X * w).array() + intercept;
    }



    Vec w;
    
    double intercept;
};



template <template <class> class Handler, class LogReg>
struct LogisticRegressionHandler : public Handler<LogReg>, LogReg
{
    template <typename... Args>
    LogisticRegressionHandler (Args&&... args) : Handler<LogReg>(std::forward<Args>(args)...) {}


    void fit (const Mat& X, const Veci& y, int K)
    {
        Handler<LogReg>::fit(X, y, K);
    }


    int predict (const Vec& x)
    {
        return Handler<LogReg>::predict(x);
    }

    Veci predict (const Mat& X)
    {
        return Handler<LogReg>::predict(X);
    }
};





#endif // CPP_ML_LOGISTIC_REGRESSION_H