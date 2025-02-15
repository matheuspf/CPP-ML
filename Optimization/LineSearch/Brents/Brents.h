#ifndef OPT_BRENTS_LS_H
#define OPT_BRENTS_LS_H

#include "../LineSearch.h"



struct Brents
{
	Brents (double tol = EPS, int maxIter = 1e4) : tol(tol), maxIter(maxIter) {}


	template <class Function>
	double operator () (Function f, double a, double x, double b)
	{
		assert(x > a && x < b && "Wrong range for search");


		int iter = 0;

		double fx, u, fu, w, fw, v, fv, d = 0.0, e = 0.0;

		w = v = x, fx = f(x), fw = fv = fx;


		// if(abs(b - x) > abs(x - a))
		// 	u = x + q * (b - x), fu = f(u);

		// else
		// {
		// 	u = x - q * (x - a), fu = f(x);
		// 	swap(x, u), swap(fx, fu);
		// }


		while(abs(b - x) > 2 * tol && abs(x - a) > 2 * tol && ++iter < maxIter)
		{
			bool s = abs(x - a) > abs(b - x);
			e = d;

			if(d > 0.0)
			{
				double p = x - w, t = x - v;
				double fp = fx - fw, ft = fx - fv;

				double num = -0.5 * (t * t * fp - p * p * ft), den = (t * fp - p * ft);


				d = num / den;

				if(!isfinite(d) || abs(d) >= 0.5 * abs(e) || x + d <= a || x + d >= b)
					d = r * (s ? a - x : b - x);

				else
					s = d < 0.0;
			}

			else
				d = q * (s ? a - x : b - x);


			u = x + d, fu = f(u);


			if(fu <= fx)
			{
				(s ? b : a) = x;

				shift(v, w, x, u);
				shift(fv, fw, fx, fu);
			}

			else
			{
				(s ? a : b) = u;

				if(fu <= fw || w == x)
					shift(v, w, u), shift(fv, fw, fu);

				else if(fu <= fv || v == x || v == w)
					v = u, fv = fu;
			}
		}

		return x;
	}




	template <class Function>
	double operator () (Function f, double a, double b)
	{
		assert(a < b && "Wrong range for search");


		double x = a + q * (b - a);
		double y = b - q * (b - a);

		double fa = f(a), fb = f(b);
		double fx = f(x), fy = f(y);


		if(fx < fy)
		{
			if(fx < fa && fx < fb)
				return operator()(f, a, x, b);

			return operator()(f, a, x, y);
		}

		else
		{
			if(fy < fa && fy < fb)
				return operator()(f, a, y, b);

			return operator()(f, x, y, b);
		}


		return x;		// Quit CS
	}



	template <class Function>
	double operator () (Function f, tuple<double, double, double> tup)
	{
		return operator()(f, get<0>(tup), get<1>(tup), get<2>(tup));
	}





	double tol;

	int maxIter;


	static constexpr double r = 1.0 / goldenRatio;
	static constexpr double q = 1.0 - r;
};



#endif // OPT_BRENTS_LS_H