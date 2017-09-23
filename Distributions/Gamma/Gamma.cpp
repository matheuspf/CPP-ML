#include "Gamma.h"
#include <gnuplot-iostream.h>


int main ()
{
	Gamma dist(100.0, 14.0);



	// Generate and fit

	int pts = 1e5;

	Vec x(pts);

	FOR(i, pts)
		x(i) = dist();

	dist.fit(x);

	DB(dist.alpha << "     " << dist.beta);





	// Plot
	Gnuplot gp;


	int it = 1e4;
	double l = 4.5, u = 10.0, d = (u - l) / it;
	
	vector<pd> v(it);

	FOR(i, it)
		v[i] = pd(l + i * d, dist(l + i * d));
		

	gp << "plot '-' with lines\n";
	gp.send1d(v);

	cin.get();



	return 0;
}