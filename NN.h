#ifndef _NN_H
#define _NN_H

#include <vector>
#include <string>
#include <iostream>
#include <ctime>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <list>
#include <tuple>

using namespace std;

class NeuralNet{
private:
    class Nodes {
    public:
        double inputN;
        double actN;
        double weightN;
        vector<double> weights;
    };

public:

    //class Nodes;
    // Function that reads wieghts files (initial neural network)
    NeuralNet(string inputFile);
    //NerualNet();

    // Function that reads the training and test data files
    void dataLoad(ifstream &inputFile);

    // Function that actually trains
    void train();

    // Function that does test
    void test();

    // Activation Function
    void activation();

    //     - sigmoid???
    double sigmoid(double d);
    double derSigmoid(double d);

    // Output file function (results file)
    //     - I think this contains all the wieght results as well as the following:
    //     - Overall accuracy
    //     - Precision
    //     - Recall
    //     - F1

    void results(string outputFile);
    double accurary;
    double precision;
    double recall;
    double f1;
    void calcMetrics(string outputFile, vector<vector<double>> CM);

    //     - Confusion Matrix 

    

    // NN layers
    int numInputNodes;
    int numHiddenNodes;
    int numOutputNodes;

    vector<Nodes> inputWieghtsLayer;
    vector<Nodes> hiddenWieghtsLayer;
    vector<Nodes> outputWieghtsLayer;

    void getExamples(string file, double b);

    void preTrain();
    string trainF;
    string outF;
    int epochs; //number of epochs
    double lr; //learning rate

    vector<vector<vector<double>>> allExamplesTrain;
    int exampleLength;

    double bTrain; //will be value used for all info in training file
    
    void preTest();
    string testF;
    string outTestF;
    vector<vector<vector<double>>> allExamplesTest;

    double bTest; //will be value used for all info in training file

    // Micro-averaging stuff
    double globA = 0;
    double globB = 0;
    double globC = 0;
    double globD = 0;
    
    void initializeCM();
    vector<vector<double>> CM;

    int flag;

};

#endif //_NN_H