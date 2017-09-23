#include "BetaMixture.h"
#include <gnuplot-iostream.h>


int main ()
{
	BetaMixture dist;

	Beta b1(1.0, 5.0), b2(5.0, 1.0);


	Vec x(int(1e4));

	generate(x.data(), x.data() + x.rows() / 2, [&]{ return b1(); });
	generate(x.data() + x.rows() / 2, x.data() + x.rows(), [&]{ return b2(); });



	dist.fit(x, 2);

	DB("\n\n");
	DB(dist.betas[0].a << "     " << dist.betas[0].b);
	DB(dist.betas[1].a << "     " << dist.betas[1].b);








	// int it = 1e3;

	// vector<pd> v;

	// FOR(i, it)
	// 	v.pb(pd(double(i) / it, b1(double(i) / it) + b2(double(i) / it)));



	// Gnuplot gp;


	// gp << "set xrange[0:1]\n";
	// gp << "plot '-' with lines\n";
	// gp.send1d(v);


	// cin.get();



	return 0;
}