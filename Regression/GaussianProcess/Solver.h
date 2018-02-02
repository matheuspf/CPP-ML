#pragma once

#include "../../Modelo.h"


struct Invert
{
    Vec operator () (const Mat& A, const Vec& b) const
    {
        Eigen::LLT<Mat> llt(A);
	
        if(llt.info() == Eigen::Success)
            return llt.solve(b);
        
        Eigen::ColPivHouseholderQR<Mat> qr(A);

        return qr.solve(b);
    }
};