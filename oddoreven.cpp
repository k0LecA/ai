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
    vector<double> h={0,0,0,0};
    double bias=0;
    double out;
    vector<double> w;
    vector<double> outw;
    vector<double> d;
    vector<double> wd;
    vector<double> outwd;
    double od;
    vector<double> in;
    double lr=0.25;
    vector<double> hbias(4, 0.0);
    double outbias = 0.0;
    in.resize(8);
    for(int i=0;i<in.size()*h.size();i++){
        w.push_back(((double)rand() / (RAND_MAX)) * 2.0 - 1.0);
    }
    for(int i=0;i<h.size();i++){
        outw.push_back(((double)rand() / (RAND_MAX)) * 2.0 - 1.0);
    }
    while(input>0){
        for(int i=0;i<h.size();i++) h[i]=0;
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


        in=decToBinary(input);
        for(int i=0;i<in.size();i++){
            cout<<in[i]<<" ";
        }
        
        cout<<endl<<w.size()<<endl;

        //for(int i=0;i<in.size()*h.size();i++){
        //    cout<<w[i]<<" ";
        //}

        for(int i=0;i<h.size();i++){
            h[i]+=hbias[i];
            for(int j=0;j<in.size();j++){
                h[i]+=in[j]*w[i*in.size()+j];
            }
            h[i]=relu(h[i]);
        }
        outbias += lr * od;
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

        if(round(out)==1)
            cout<<endl<<"odd!!!";
        else if(round(out)==0)
            cout<<endl<<"even!!!";
        else
            cout<<"somewhere is error";

        od=out*(1-out)*error;

        d.clear();
        d.resize(h.size());
        for(int i=0;i<h.size();i++)
        {
            //d[i]=h[i]*(1-h[i])*(outw[i]*od);
            d[i] = (h[i] > 0 ? 1.0 : 0.0) * (outw[i] * od);
        }

        outwd.clear();
        outwd.resize(outw.size());
        for(int i=0;i<outw.size();i++)
        {
            outwd[i]=lr*od*h[i];
            outw[i]+=outwd[i];
        }

        wd.clear();
        wd.resize(w.size());
        for(int i=0; i<h.size(); i++)
        {
            hbias[i] += lr * d[i];
            for(int j=0; j<in.size(); j++)
            {
                int weightIndex = i * in.size() + j;
                w[weightIndex] += lr * d[i] * in[j];
            }
        }
    }
}