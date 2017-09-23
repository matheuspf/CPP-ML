#include "Multinomial.h"
#include <gnuplot-iostream.h>


int main ()
{
	int N = 5;

	Vec x(N);

	fill(x.data(), x.data() + x.rows() - 1, 1.5 / 8);
	x(x.rows() - 1) = 1.0 - 6.0 / 8;


	Multinomial dist(x);

	DB(dist.mean());



	vector<int> v;

	FOR(i, 1e5) v.pb(dist());




	Gnuplot gp;


	gp << "min = 0\n";
	gp << "max = 5\n";
	gp << "n = 5\n";
	gp << "width =(max - min) / n\n";
	gp << "hist(x,width)=width*floor(x/width)+width/2.0\n";
	//gp << "set style fill solid 1.0\n";
	gp << "plot '-' u (hist($1,width)):(1.0) smooth freq w boxes lc rgb\"red\" notitle\n";
	gp.send1d(v);

	cin.get();


	return 0;
}