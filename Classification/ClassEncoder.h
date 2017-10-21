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
            Veci y_enc = lenc.fitTransform(y);
            numClasses = lenc.numClasses;

            return static_cast<Impl&>(*this).fit_(X, y_enc);
        }
            
        return static_cast<Impl&>(*this).fit_(X, y);
    }


    auto predict (const Vec& x)
    {
        auto label = static_cast<Impl&>(*this).predict_(x);

        if(encodeLabels)
            label = lenc.reverseMap[label];
        
        return label;
    }

    auto predict (const Mat& X)
    {
        auto labels = static_cast<Impl&>(*this).predict_(X);

        if(encodeLabels)
        {
            std::transform(std::begin(labels), std::end(labels), std::begin(labels), [&](const auto& label)
            {
                return lenc.reverseMap[label];
            });
        }

        return labels;
    }


    LabelEncoder<int> lenc;

    int numClasses;

    bool encodeLabels;



private:

    ClassEncoder () {}

    friend Impl;
};



template <class Impl>
struct ClassEncoder<Impl, false>
{
    decltype(auto) fit (const Mat& X, const Veci& y)
    {
        return static_cast<Impl&>(*this).fit_(X, y);
    }


    auto predict (const Vec& x)
    {
        return static_cast<Impl&>(*this).predict_(x);
    }

    auto predict (const Mat& X)
    {
        return static_cast<Impl&>(*this).predict_(X);
    }


private:

    ClassEncoder () {}
    
    friend Impl;
};






#endif // CPP_ML_CLASS_ENCODER_H