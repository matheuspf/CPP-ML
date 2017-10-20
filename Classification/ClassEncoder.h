#ifndef CPP_ML_CLASS_ENCODER_H
#define CPP_ML_CLASS_ENCODER_H

#include "../Modelo.h"

#include "../Preprocessing/Preprocess.h"



template <class Impl, bool Encode>
struct ClassEncoder
{
    decltype(auto) fit (const Mat& X, const Veci& y, int numClasses_ = 0)
    {
        numClasses = numClasses_;
        encodeLabels = numClasses <= 0;

        if(encodeLabels)
        {
            const_cast<Veci&>(y) = lenc.fitTransform(y);
            numClasses = lenc.numClasses;
        }

        if(encode)
            

        return static_cast<Impl&>(*this).fit(X, y);
    }


    auto predict (const Vec& x)
    {
        auto label = static_cast<Impl&>(*this).predict(x);

        if(encodeLabels)
            label = lenc.reverseMap[label];
        
        return label;
    }

    auto predict (const Mat& X)
    {
        auto labels = static_cast<Impl&>(*this).predict(X);

        if(encodeLabels)
        {
            std::transform(std::begin(labels), std::end(labels), std::begin(labels), [&](const auto& label)
            {
                return lenc.reverseMap[label];
            });
        }

        return labels;
    }


    LabelEncoder lenc;

    int numClasses;

    bool encodeLabels;



private:

    ClassEncoder () {}

    friend Impl;
};



template <class Impl>
struct ClassEncoder<Impl, false>
{
    decltype(auto) fit (const Mat& X, const Veci& y, int)
    {
        return static_cast<Impl&>(*this).fit(X, y);
    }


    auto predict (const Vec& x)
    {
        return static_cast<Impl&>(*this).predict(x);
    }

    auto predict (const Mat& X)
    {
        return static_cast<Impl&>(*this).predict(X);
    }


private:

    ClassEncoder () {}
    
    friend Impl;
};






#endif // CPP_ML_CLASS_ENCODER_H