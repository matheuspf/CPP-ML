#include "BernoulliBayes.h"
#include <gnuplot-iostream.h>


int main ()
{
	BernoulliBayes dist(2.0, 2.0);


	Veci data = Veci::Constant(10, 1);

	generate(&data(0), &data(0)+data.rows() / 4, []{ return 0; });


	dist.fit(data);



	vi v;

	FOR(i, 10000) v.pb(dist());


	DB(dist.mean());


	Gnuplot gp;



	gp << "min = 0\n";
	gp << "max = 2\n";
	gp << "n = 2\n";
	gp << "width =(max - min) / n\n";
	gp << "hist(x,width)=width*floor(x/width)+width/2.0\n";
	gp << "plot '-' u (hist($1,width)):(1.0) smooth freq w boxes lc rgb\"red\" notitle\n";
	
	gp.send1d(v);


	return 0;
}