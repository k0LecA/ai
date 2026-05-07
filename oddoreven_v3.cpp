#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <sstream>
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
    for (int i = 0; i < (int)h.size(); i++) {
        h[i] = hbias[i];
        for (int j = 0; j < (int)in.size(); j++) {
            h[i] += in[j] * w[i * in.size() + j];
        }
        h[i] = relu(h[i]);
    }
}

double forwardOutput(const vector<double>& h, const vector<double>& outw, double outbias) {
    double out = outbias;
    for (int i = 0; i < (int)h.size(); i++) {
        out += h[i] * outw[i];
    }
    return sigmoid(out);
}

double computeOd(double out, double error) {
    return out * (1 - out) * error;
}

vector<double> computeHiddenDeltas(const vector<double>& h, const vector<double>& outw, double od) {
    vector<double> d(h.size());
    for (int i = 0; i < (int)h.size(); i++) {
        d[i] = (h[i] > 0 ? 1.0 : 0.0) * (outw[i] * od);
    }
    return d;
}

void updateOutputWeights(vector<double>& outw, double& outbias,
                         const vector<double>& h, double od, double lr) {
    outbias += lr * od;
    for (int i = 0; i < (int)outw.size(); i++) {
        outw[i] += lr * od * h[i];
    }
}

void updateHiddenWeights(vector<double>& w, vector<double>& hbias,
                         const vector<double>& d, const vector<double>& in,
                         int hSize, double lr) {
    for (int i = 0; i < hSize; i++) {
        hbias[i] += lr * d[i];
        for (int j = 0; j < (int)in.size(); j++) {
            int weightIndex = i * in.size() + j;
            w[weightIndex] += lr * d[i] * in[j];
        }
    }
}

void trainSample(int num, double target,
                 vector<double>& h, vector<double>& w, vector<double>& outw,
                 vector<double>& hbias, double& outbias, double& od, double lr) {
    fill(h.begin(), h.end(), 0.0);

    vector<double> in = decToBinary(num);
    forwardHidden(h, in, w, hbias);

    outbias += lr * od;
    double out = forwardOutput(h, outw, outbias);

    double error = target - out;
    od = computeOd(out, error);

    vector<double> d = computeHiddenDeltas(h, outw, od);
    updateOutputWeights(outw, outbias, h, od, lr);
    updateHiddenWeights(w, hbias, d, in, h.size(), lr);
}

bool loadTrainingData(const string& filename, vector<int>& trainNums, vector<double>& trainTargets) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Error: could not open file '" << filename << "'" << endl;
        return false;
    }
    int num;
    double target;
    while (file >> num >> target) {
        if (num < 1 || num > 255) {
            cout << "Skipping out-of-range value: " << num << endl;
            continue;
        }
        trainNums.push_back(num);
        trainTargets.push_back(target);
    }
    file.close();
    if (trainNums.empty()) {
        cout << "Error: no valid training data found in file." << endl;
        return false;
    }
    return true;
}

void trainEpochs(int epochs, const vector<int>& trainNums, const vector<double>& trainTargets,
                 vector<double>& h, vector<double>& w, vector<double>& outw,
                 vector<double>& hbias, double& outbias, double lr) {
    cout << "Training on " << trainNums.size() << " samples for " << epochs << " epochs..." << endl;
    for (int ep = 0; ep < epochs; ep++) {
        double od = 0.0;
        double totalError = 0.0;
        for (int s = 0; s < (int)trainNums.size(); s++) {
            fill(h.begin(), h.end(), 0.0);
            vector<double> in = decToBinary(trainNums[s]);
            forwardHidden(h, in, w, hbias);
            outbias += lr * od;
            double out = forwardOutput(h, outw, outbias);
            double error = trainTargets[s] - out;
            totalError += error * error;
            od = computeOd(out, error);
            vector<double> d = computeHiddenDeltas(h, outw, od);
            updateOutputWeights(outw, outbias, h, od, lr);
            updateHiddenWeights(w, hbias, d, in, h.size(), lr);
        }
        if ((ep + 1) % 100 == 0 || ep == 0) {
            cout << "Epoch " << ep + 1 << " / " << epochs
                 << "  MSE: " << totalError / trainNums.size() << endl;
        }
    }
    cout << "Training complete." << endl << endl;
}

void printBinary(const vector<double>& in) {
    for (int i = 0; i < (int)in.size(); i++) {
        cout << in[i] << " ";
    }
    cout << endl;
}

void printResults(double target, double out, double error) {
    cout << "Expected result: " << target << endl;
    cout << "Result:          " << out << endl;
    cout << "Error:           " << error << endl;
}

void printParity(double out) {
    if (round(out) == 1)
        cout << "=> odd!!!" << endl;
    else if (round(out) == 0)
        cout << "=> even!!!" << endl;
    else
        cout << "=> somewhere is error" << endl;
}

int readInput() {
    int input;
    cout << endl << "Input from 1 to 255 (0 to quit): ";
    cin >> input;
    return input;
}

bool validateInput(int input) {
    if (input == 0) return false;
    if (input > 255 || input < 1) {
        cout << "Wrong input" << endl;
        return false;
    }
    return true;
}

void runInferenceLoop(vector<double>& h, vector<double>& w, vector<double>& outw,
                      vector<double>& hbias, double& outbias) {
    int input = 1;
    while (true) {
        input = readInput();
        if (input == 0) break;
        if (!validateInput(input)) continue;

        fill(h.begin(), h.end(), 0.0);
        vector<double> in = decToBinary(input);
        printBinary(in);

        forwardHidden(h, in, w, hbias);
        double out = forwardOutput(h, outw, outbias);

        double target = input % 2;
        double error = target - out;
        printResults(target, out, error);
        printParity(out);
    }
}

int main() {
    srand((unsigned int)time(0));

    string filename="data.txt";
    cout << "Training data file: "<<filename<<endl;


    vector<int> trainNums;
    vector<double> trainTargets;
    if (!loadTrainingData(filename, trainNums, trainTargets)) return 1;

    int epochs;
    cout << "Number of epochs: ";
    cin >> epochs;

    vector<double> h(4, 0.0);
    double lr = 0.25;
    vector<double> hbias(4, 0.0);
    double outbias = 0.0;

    vector<double> in(8, 0.0);
    vector<double> w = initWeights(in.size() * h.size());
    vector<double> outw = initWeights(h.size());

    trainEpochs(epochs, trainNums, trainTargets, h, w, outw, hbias, outbias, lr);

    cout << "--- Inference mode (0 to quit) ---" << endl;
    runInferenceLoop(h, w, outw, hbias, outbias);

    return 0;
}