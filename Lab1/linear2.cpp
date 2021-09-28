#include <iostream>
#include <cmath>
#include <time.h>
#include <iomanip>
using namespace std;

const int N=64*256*1024*4;

int main()
{
    double flag=1; double outcome=0;
    double begin=clock();
    for(int i=1;i<=N;i++){
        outcome+=flag*(1.0/(2*i-1));
        flag*=-1;
    }
    outcome*=4;
    double end=clock();
    cout<<"linear2: "<<"N="<<N<<" PI="<<fixed<<setprecision(10)<<outcome<<" time spend ="<<(end-begin)/CLOCKS_PER_SEC<<endl;
    return 0;
}
