#include <fstream>
#include <ctime>
#include <cstdlib>
using namespace std;
int main(){
    ofstream fw("data.txt");
    int num=0;
    for(int i=0;i<10000;i++)
    {
        num=((double)rand() / (RAND_MAX)) * 255 - 1;
        fw<<num<<" "<<num%2<<endl;
    }
    fw.close();
}