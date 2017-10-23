#ifndef CPP_ML_CLASS_ENCODER_H
#define CPP_ML_CLASS_ENCODER_H

#include "../Modelo.h"

#include "../Preprocessing/Preprocess.h"


HAS_VAR(classLabels);


template <class Classifier>
struct ClassEncoder
{
    decltype(auto) fit (const Mat& X, const Veci& y, int numClasses_ = 0)
    {
        numClasses = numClasses_;
        encodeLabels = numClasses <= 0;

        if(encodeLabels)
        {
            Veci y_enc = encode(y);
            numClasses = lenc.numClasses;

            return static_cast<Classifier&>(*this).fit_(X, y_enc);
        }
            
        return static_cast<Classifier&>(*this).fit_(X, y);
    }


    

    template <class T = Classifier, std::enable_if_t<has_classLabels<T>(), int> = 0>
    Veci encode (const Veci& y)
    {
        return lenc.fitTransform(y, Classifier::classLabels);
    }

    template <class T = Classifier, std::enable_if_t<!has_classLabels<T>(), int> = 0>
    Veci encode (const Veci& y)
    {
        return lenc.fitTransform(y);
    }




    auto predict (const Vec& x)
    {
        auto label = static_cast<Classifier&>(*this).predict_(x);

        if(encodeLabels)
            label = lenc.reverseMap[label];
        
        return label;
    }

    auto predict (const Mat& X)
    {
        auto labels = static_cast<Classifier&>(*this).predict_(X);

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

    friend Classifier;
};





#endif // CPP_ML_CLASS_ENCODER_H