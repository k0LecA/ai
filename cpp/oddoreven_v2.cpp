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

vector<double> initWeights(int count) {
    vector<double> weights;
    for (int i = 0; i < count; i++) {
        weights.push_back(((double)rand() / (RAND_MAX)) * 2.0 - 1.0);
    }
    return weights;
}

void forwardHidden(vector<double>& h, const vector<double>& in,
                   const vector<double>& w, const vector<double>& hbias) {
    for (int i = 0; i < h.size(); i++) {
        h[i] = hbias[i];
        for (int j = 0; j < in.size(); j++) {
            h[i] += in[j] * w[i * in.size() + j];
        }
        h[i] = relu(h[i]);
    }
}

double forwardOutput(const vector<double>& h, const vector<double>& outw, double outbias) {
    double out = outbias;
    for (int i = 0; i < h.size(); i++) {
        out += h[i] * outw[i];
    }
    return sigmoid(out);
}

double computeError(double target, double out) {
    return target - out;
}

double computeOd(double out, double error) {
    return out * (1 - out) * error;
}

vector<double> computeHiddenDeltas(const vector<double>& h, const vector<double>& outw, double od) {
    vector<double> d(h.size());
    for (int i = 0; i < h.size(); i++) {
        d[i] = (h[i] > 0 ? 1.0 : 0.0) * (outw[i] * od);
    }
    return d;
}

void updateOutputWeights(vector<double>& outw, double& outbias,
                         const vector<double>& h, double od, double lr) {
    outbias += lr * od;
    for (int i = 0; i < outw.size(); i++) {
        outw[i] += lr * od * h[i];
    }
}

void updateHiddenWeights(vector<double>& w, vector<double>& hbias,
                         const vector<double>& d, const vector<double>& in,
                         int hSize, double lr) {
    for (int i = 0; i < hSize; i++) {
        hbias[i] += lr * d[i];
        for (int j = 0; j < in.size(); j++) {
            int weightIndex = i * in.size() + j;
            w[weightIndex] += lr * d[i] * in[j];
        }
    }
}

int readInput() {
    int input;
    cout << endl << "Input from 1 to 255: ";
    cin >> input;
    return input;
}

bool validateInput(int input) {
    if (input == -1) return false;
    if (input > 255 || input < 1) {
        cout << endl << "Wrong input";
        return false;
    }
    return true;
}

void printBinary(const vector<double>& in) {
    for (int i = 0; i < in.size(); i++) {
        cout << in[i] << " ";
    }
    cout << endl;
}

void printResults(double target, double out, double error) {
    cout << endl << "expected result: " << target << endl;
    cout << "result: " << out << endl;
    cout << "error: " << error << endl;
}

void printParity(double out) {
    if (round(out) == 1)
        cout << endl << "odd!!!";
    else if (round(out) == 0)
        cout << endl << "even!!!";
    else
        cout << "somewhere is error";
}

int main() {
    int input = 1;
    vector<double> h = {0, 0, 0, 0};
    double out;
    double od = 0;
    vector<double> in;
    double lr = 0.25;
    vector<double> hbias(4, 0.0);
    double outbias = 0.0;

    in.resize(8);
    vector<double> w = initWeights(in.size() * h.size());
    vector<double> outw = initWeights(h.size());

    while (input > 0) {
        fill(h.begin(), h.end(), 0);

        input = readInput();

        if (input == -1) continue;
        if (!validateInput(input)) {
            input = 1;
            continue;
        }

        in = decToBinary(input);
        printBinary(in);
        cout << w.size() << endl;

        forwardHidden(h, in, w, hbias);

        outbias += lr * od;
        out = forwardOutput(h, outw, outbias);

        double target = input % 2;
        double error = computeError(target, out);
        printResults(target, out, error);
        printParity(out);

        od = computeOd(out, error);
        vector<double> d = computeHiddenDeltas(h, outw, od);

        updateOutputWeights(outw, outbias, h, od, lr);
        updateHiddenWeights(w, hbias, d, in, h.size(), lr);
    }
}