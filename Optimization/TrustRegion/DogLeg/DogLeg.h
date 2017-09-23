#ifndef OPT_TR_CAUCHY_POINT_H
#define OPT_TR_CAUCHY_POINT_H

#include "../TrustRegion.h"


struct DogLeg : public TrustRegion<DogLeg>
{
	template <class Function, class Gradient, class Hessian>
	Vec direction (Function function, Gradient gradient, Hessian hessian, Vec x, 
				   double delta, double fx, const Vec& gx, Mat hx)
	{
		Vec pb = -hx.ldlt().solve(gx);
		//Vec pb = -hx.colPivHouseholderQr().solve(gx);


		if(pb.norm() <= delta)
			return pb;

		Vec pu = -(gx.dot(gx) / (gx.transpose() * hx * gx)) * gx;

		Vec diff = (pb - pu);


		double a = diff.squaredNorm();

		double b = 2 * (pu.dot(diff) - a);

		double c = a - pow(delta, 2) - 2 * pu.dot(diff) + pu.dot(pu);


		double tau = -(b + sqrt(b*b - 4 * a * c)) / (2 * a);

		return pu + (tau - 1.0) * diff;
	}


	template <class Function, class Gradient, class Hessian>
	Vec direction (Function function, Gradient gradient, Hessian hessian, Vec x, double delta)
	{
		return this->operator()(function, gradient, hessian, x, delta, function(x), gradient(x), hessian(x));
	}
};




#endif // OPT_TR_CAUCHY_POINT_H