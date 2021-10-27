#include <iostream>
#include <string.h>
#include <time.h>
#include <iomanip>

using namespace std;

const int m=32;
const int N=4096;

int main()
{
    double martix[m][m];
    double temp[m][m];
    double temp2[m][m];
    // cout<<"elem value is "<<1.0/m<<endl;
    //init martix
    for(int i=0;i<m;i++){
        for(int j=0;j<m;j++) {martix[i][j]=1.0/(m);temp[i][j]=1.0/(m);temp2[i][j]=0;}
    }
    double begin=clock();
    for(int exp=0;exp<N;exp++){
        for(int i=0;i<m;i++){
            for(int j=0;j<m;j++){
                for(int k=0;k<m;k++){
                    temp2[i][j]+=temp[i][k]*martix[k][j];
                }
            }
        }
        memcpy(temp,temp2,sizeof(double)*m*m);
        memset(temp2,0,sizeof(double)*m*m);
    }
    memcpy(martix,temp,sizeof(double)*m*m);
    double end=clock();
    cout<<"Linear's timespend is "<<fixed<<setprecision(10)<<(end-begin)/CLOCKS_PER_SEC<<endl;
    // cout<<"After exp N, martix's value = "<<martix[0][0]<<endl;
    return 0;
}