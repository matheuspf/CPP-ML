#include "CG.h"


struct Rosenbrock
{
	double operator () (const Vec& x) const
	{
		double r = 0.0;

        for(int i = 0; i < x.rows() - 1; ++i)
        	r += 100.0 * pow(x(i+1) - pow(x(i), 2), 2) + pow(x(i) - 1.0, 2);

        return r;
	}
};





int main ()
{
	CG<PR_FR> cg;

	Vec x = Vec::Constant(1000, 5.0);

	x = cg(Rosenbrock(), x);


	DB(x.transpose());




	return 0;
}