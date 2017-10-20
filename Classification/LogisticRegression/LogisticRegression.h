#ifndef CPP_ML_LOGISTIC_REGRESSION_H
#define CPP_ML_LOGISTIC_REGRESSION_H

#include "../../Modelo.h"

#include "../../Preprocessing/Preprocess.h"

#include "../../Optimization/Newton/Newton.h"

#include "../../Regularizers.h"

#include "../OVA/OVA.h"

//#include "../OVO/OVO.h"



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

            // else if(multiClassType == "OVO")
            //     impl = new LogisticRegressionHandler<OVO, LogisticRegressionTwoClass<Regularizer, Optimizer>>(alpha, optimizer);
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

    Vec predictProb (const Mat& X)
    {
        return impl->predictProb(X);
    }


    // double predictMargin (const Vec& x)
    // {
    //     return return impl->predictMargin(x);
    // }

    Vec predictMargin (const Mat& X)
    {
        return impl->predictMargin(X);
    }
    

    
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

    //virtual double predictProb (const Vec&) = 0;
    virtual Vec predictProb (const Mat&)    = 0;
    
    //virtual double predictMargin (const Vec&) = 0;    
    virtual Vec predictMargin (const Mat&)    = 0; 


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








#endif // CPP_ML_LOGISTIC_REGRESSION_H