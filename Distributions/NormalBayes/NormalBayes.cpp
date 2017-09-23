#include "NormalBayes.h"
#include <gnuplot-iostream.h>


int main ()
{
	Normal normal(0.0, 1.0);

	NormalPost normPost(1.0, 1.0, 1.0, 0.0);

	NormalBayes normBayes(1.0, 1.0, 1.0, 0.0);



	int m = 10;

	Vec v(m);

	FOR(i, m)
		v(i) = normal();

	normPost.fit(v);
	normBayes.fit(v);


	int t = 1e3;
	double l = -4.0, u = 4.0, d = (u - l) / t;

	vector<pd> vp(t), vb(t);

	FOR(i, t)
	{
		double x = l + i * d;

		vp[i] = pd(x, normPost(x));
		vb[i] = pd(x, normBayes(x));
	}

	DB(normBayes.mode() << "      " << normBayes(normBayes.mode()));



	Gnuplot gp;

	gp << "plot '-' with lines title 'Posteriori', "
	   << "'-' with lines title 'Bayesian'\n";
	gp.send1d(vp);
	gp.send1d(vb);

	cin.get();


	return 0;
}