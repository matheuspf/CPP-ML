#include "LinearRegression.h"



int main ()
{
	Matrix A(3, 2);
	Vector b(3);

	A << 1, 2, 2, 4, 3, 7;
	b << 10, 2, 1;


	LinearRegression lr(1e1);

	lr.learn(A, b);

	DB(lr(A) << "\n\n");

	DB(lr.phi << "    " << lr.bias);



	return 0;
}