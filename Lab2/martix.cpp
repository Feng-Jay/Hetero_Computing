#include <iostream>
#include <string.h>

using namespace std;

const int m=4;
const int N=4;

int main()
{
    double martix[m][m];
    double temp[m][m];
    //init martix
    for(int i=0;i<m;i++){
        for(int j=0;j<m;j++) {martix[i][j]=1.0/(m);temp[i][j]=0;}
    }
    for(int exp=0;exp<N/2;exp++){
        for(int i=0;i<m;i++){
            for(int j=0;j<m;j++){
                for(int k=0;k<m;k++){
                    temp[i][j]+=martix[i][k]*martix[k][j];
                }
            }
        }
        memcpy(martix,temp,sizeof(double)*m*m);
        memset(temp,0,sizeof(double)*m*m);
    }

    for(int i=0;i<m;i++){
        for(int j=0;j<m;j++)
        cout<<martix[i][j]<<" ";
        cout<<endl;
    }
}