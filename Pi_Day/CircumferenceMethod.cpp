#include<iostream>
#include<cmath>
#include<iomanip>
using namespace std;
class cyclotomic {
public:
	void cyclotomic1()
	{
		int i = 1,a=1;
		double len = 1;
		while(i<=n)
		{ 
			len = 2 - sqrt(4 - len);   //内接多边形的边长
			a *= 2;
			i++;
		}
		pi = 3.0 * (double)a * sqrt(len);
	}
	void showresult()
	{
		cout.precision(12);   //小数点后第十二位
		cout << pi << endl;
	}
	int n;
	double pi;
};
void text()
{
	cyclotomic c;
	cout << "输入要切割的次数：" << endl;
	cin>>c.n;
	c.cyclotomic1();
	c.showresult();
}
int main()
{
	text();
}
