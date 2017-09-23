#include "Beta.h"
#include <gnuplot-iostream.h>


int main ()
{
	Beta dist(0.01, 7.0);



	// int it = 100;

	// vector<pair<double, double>> v;

	// FOR(i, it)
	// 	v.emplace_back(double(i) / it, dist(double(i) / it));

	// Gnuplot gp;

	// gp << "plot '-' using 1:2 with lines\n";

	// gp.send1d(v);




	Vec x(int(1e6));

	generate(&x(0), &x(x.rows()-1)+1, [&]{ return dist(); });

	dist.fit(x);

	DB(dist.a << "     " << dist.b);




	// int it = 1e6, n = 50;

	// vector<int> v;

	// FOR(i, it)
	// 	v.pb(llround(n*dist()));


	// gp << "min = 0\n";
	// gp << "max = 50\n";
	// gp << "n = 50\n";
	// gp << "width =(max - min) / n\n";
	// gp << "hist(x,width)=width*floor(x/width)+width/2.0\n";
	// gp << "plot '-' u (hist($1,width)):(1.0) smooth freq w boxes lc rgb\"red\" notitle\n";
	// gp.send1d(v);





	// DB(dist.mean() << "      " << dist.variance());


	// int it = 1e4;

	// vector<pd> v;

	// FOR(i, it)
	// 	v.pb(pd(double(i) / it, dist(double(i) / it)));


	// gp << "set xrange[0:1]\n";
	// gp << "plot '-' with lines\n";
	// gp.send1d(v);



	// cin.get();



	return 0;
}