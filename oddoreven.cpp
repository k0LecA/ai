#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

vector<double> decToBinary(int n) {
    const int BITS = 8;
    vector<double> bin(BITS, 0.0);
    
    for (int i = BITS - 1; i >= 0 && n > 0; i--) {
        bin[i] = n % 2;
        n /= 2;
    }
    return bin;
}

double relu(double x) { return x > 0 ? x : 0; }

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

int main(){

    int input=1;

    while(input>0){
        cout<<endl<<"Input from 1 to 255: ";
        cin>>input;

        if(input==-1){
            continue;
        }
        if(input>255 || input<1) { 
            cout<<endl<<"Wrong input";
            input=1;
            continue;
        }


        vector<double> in=decToBinary(input);
        for(int i=0;i<in.size();i++){
            cout<<in[i]<<" ";
        }
        

        vector<double> h={0,0,0,0};
        double bias=0;
        vector<double> w;
        for(int i=0;i<in.size()*h.size();i++){
            w.push_back(((double)rand() / (RAND_MAX)) * 2.0 - 1.0);
        }
        cout<<endl<<w.size()<<endl;

        //for(int i=0;i<in.size()*h.size();i++){
        //    cout<<w[i]<<" ";
        //}

        for(int i=0;i<h.size();i++){
            h[i]+=bias;
            for(int j=0;j<in.size();j++){
                h[i]+=in[j]*w[i*in.size()+j];
            }
            h[i]=relu(h[i]);
        }

        double outbias=0;
        double out=0;
        vector<double> outw;
        for(int i=0;i<h.size();i++){
            outw.push_back(((double)rand() / (RAND_MAX)) * 2.0 - 1.0);
        }

        out=outbias;
        for(int i=0;i<h.size();i++){
            out+=h[i]*outw[i];
        }
        out=sigmoid(out);
        double target=input%2;
        cout<<endl<<"expected result: "<<target<<endl;
        cout<<"result: "<<out<<endl;

        double error=target-out;
        cout<<"error: "<<error<<endl;
    }
}