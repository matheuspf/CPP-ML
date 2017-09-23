#include "NormalPost.h"
#include <gnuplot-iostream.h>


int main ()
{
	Normal nm(-1.0, 1.0);
	NormalPost dist(1.0, 1.0, 1.0, 0.0);


	DB(dist.mu << "      " << dist.sigma);

	int it = 1e6;

	Vec x(it);

	FOR(i, it)
		x(i) = dist();


	dist.fit(x);

	DB(dist.mu << "      " << dist.sigma);




	// double it = 1e3, l = -4, u = 4, d = (u - l) / it;

	// vector<pd> v(it);

	// FOR(i, it)
	// 	v[i] = pd(l + i * d, dist(l + i * d));


	// Gnuplot gp;


	// gp << "plot '-' with lines\n";
	// gp.send1d(v);


	// cin.get();


	return 0;
}