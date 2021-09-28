#include <iostream>
#include <cmath>
#include <iomanip>
#include <time.h>

using namespace std;

const int N=64*256*1024*4;

int main()
{
    double outcome=0;
    double begin=clock();
    for(int i=0;i<N;i++) outcome+=4.0/(1+(i+0.5)/(N)*(i+0.5)/(N));
    outcome/=N;
    double end=clock();
    cout<<"linear1: "<<"N="<<N<<" PI="<<fixed<<setprecision(10)<<outcome<<"time spend ="<<(end-begin)/CLOCKS_PER_SEC<<endl;
    return 0;
}
