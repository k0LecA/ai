# ai

This is a learning project. It is meant for understanding how a simple neural network works from scratch, not for production use.

The network learns to classify numbers from 1 to 255 as odd or even. It takes an 8-bit binary representation of a number as input, passes it through a hidden layer, and outputs a value close to 0 (even) or 1 (odd). Training is done with backpropagation and gradient descent.

## files

generatedata.cpp - generates a training data file with 10000 random numbers and their odd/even labels. output is data.txt.

data.txt - example training data file. each line has a number followed by 0 if even or 1 if odd.

oddoreven.cpp - the original version. you enter numbers manually one by one and the network updates its weights after each input.

oddoreven_v2.cpp - refactored version of oddoreven.cpp. same logic, split into functions.

oddoreven_v3.cpp - version that reads training data from a file first, trains for a number of epochs, then lets you test it by entering numbers manually.

oddoreven.py - python version of the same concept.

a.out - compiled binary, can be ignored.

## how to run

generate training data:

```
g++ generatedata.cpp -o generatedata
./generatedata
```

compile and run the file-based version:

```
g++ oddoreven_v3.cpp -o oddoreven_v3
./oddoreven_v3
```

it will ask for the training data file name (data.txt) and how many epochs to train for. after training is done you can enter numbers and see what the network predicts.

## network structure

- input layer: 8 neurons (one per bit)
- hidden layer: 4 neurons with ReLU activation
- output layer: 1 neuron with sigmoid activation
- learning rate: 0.25
- loss: mean squared error

## what this is for

this project is for learning the basics of how neural networks work. it covers forward pass, backpropagation, weight updates, and training on labeled data. the problem it solves (odd or even) is simple enough that you can follow what the network is doing step by step.