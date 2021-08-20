#include <vector>
#include <string>
#include <iostream>
#include <ctime>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <list>
#include <tuple>
#include <time.h>
#include <cmath>
#include <iomanip>
#include "NN.h"

using namespace std;

NeuralNet::NeuralNet(string inputFile){
    //open inputFile
    ifstream inputFileStream(inputFile); 

    // check if file exists
    while(!inputFileStream.is_open()){
        cout << "ERROR, INVALID FILE! Please enter an existing file:\n" << endl;
        cin >> inputFile;
        inputFileStream.open(inputFile);
        //ifstream inputFileStream(inputFile); 
    }

    //read in values
    inputFileStream >> numInputNodes >> numHiddenNodes >> numOutputNodes;

    // cout << "number of inputs nodes: " << numInputNodes << endl;
    // cout << "number of hidden nodes: " << numHiddenNodes << endl;
    // cout << "number of output nodes: " << numOutputNodes << endl;

    Nodes n;
    double w;

    // Input
    for(int i = 0; i < numInputNodes; i++){
        inputWieghtsLayer.push_back(n);
    }

    // Hidden
    for(int i = 0; i < numHiddenNodes; i++){
        hiddenWieghtsLayer.push_back(n);
        
        // Add wieghts to each of the nodes
        inputFileStream >> hiddenWieghtsLayer[i].weightN;

        for(int j = 0; j < numInputNodes; j++){
            inputFileStream >> w;
            hiddenWieghtsLayer[i].weights.push_back(w);
        }
    }

    // Output
    for(int i = 0; i < numOutputNodes; i++){
        outputWieghtsLayer.push_back(n);

        inputFileStream >> outputWieghtsLayer[i].weightN;

        for(int j = 0; j < numHiddenNodes; j++){
            inputFileStream >> w;
            outputWieghtsLayer[i].weights.push_back(w);
        }
    }

    //close input file
    inputFileStream.close();
}


double NeuralNet::sigmoid(double d){
    double result = 1 / (1 + exp(-d));
    //cout<<"working\n";
    return result;
}

double NeuralNet::derSigmoid(double d){
    double result = sigmoid(d) * (1 - sigmoid(d));
    return result;
}

void NeuralNet::results(string outputFile){
    ofstream of(outputFile);

    of << numInputNodes << " " << numHiddenNodes << " " << numOutputNodes << endl;

    // Need to print out all of the weights in the output file

    // Input -> Hidden
    for(int i = 0; i < numHiddenNodes; i++){
        of << fixed << setprecision(3) << hiddenWieghtsLayer[i].weightN;
        for(int j = 0; j < numInputNodes; j++){
            of << " " << fixed << setprecision(3) << hiddenWieghtsLayer[i].weights[j];
        }
        of << endl;
    }

    // Hidden -> Output
    for(int i = 0; i < numOutputNodes; i++){
        of << fixed << setprecision(3) << outputWieghtsLayer[i].weightN;
        for(int j = 0; j < numHiddenNodes; j++){
            of << " " << fixed << setprecision(3) << outputWieghtsLayer[i].weights[j];
        }
        of << endl;
    }

    of.close();

}

void NeuralNet::preTrain(){

    cout << "Enter training file: \n";
    cin >> trainF;

    cout << "Enter train output file: \n";
    cin >> outF;

    cout << "How many epochs?\n";
    cin >> epochs;

    cout << "What learning rate?\n";
    cin >> lr;

    getExamples(trainF, bTrain);

    
}

void NeuralNet::getExamples(string givenFile, double b){
    ifstream file(givenFile);

    // Check if given file exists
    while(!file.is_open()){
        cout << "ERROR, INVALID FILE! Please enter an existing file:\n" << endl;
        cin >> givenFile;
        file.open(givenFile);
        //ifstream file(givenFile);
    }


    file >> exampleLength;
    // cout << exampleLength << endl;
    file.ignore(256, '\n');
    
    vector<vector<double>> examples;
    //vector<vector<vector<double>>> allExamples;
    vector<double> inEx;
    vector<double> outEx;

    for(int i = 0; i < exampleLength; i++){
        
        examples.clear();
        inEx.clear();
        outEx.clear();

        for(int j = 0; j < numInputNodes; j++){
            file >> b;
            //cout << b << endl;
            inEx.push_back(b);
        }

        for(int k = 0; k < numOutputNodes; k++){
            file >> b;
            //cout << b << endl;
            outEx.push_back(b);
        }

        examples.push_back(inEx);
        examples.push_back(outEx);
        if (flag == 0){
            allExamplesTrain.push_back(examples);
        } else {
            allExamplesTest.push_back(examples);
        }
    }

    file.close();
}

void NeuralNet::train(){
    // take in given files and create all examples vector
    preTrain();
    //cout << "epochs: " << epochs << endl;

    //cout << exampleLength << endl;

    for(int i = 0; i < epochs; i++){

        for(int j = 0; j < exampleLength; j++){
            vector<double> hidErr;
            vector<double> outErr;            
            double s;
            // Propagate the inputs foward to compute the outputs
            //cout << j << endl;
            for(int k = 0; k < numInputNodes; k++){
                //cout << k << endl;
                inputWieghtsLayer[k].actN = allExamplesTrain[j][0][k];
                inputWieghtsLayer[k].inputN = allExamplesTrain[j][0][k];
            }
            
            // Hidden layer prop
            for(int k1 = 0; k1 < numHiddenNodes; k1++){
                s = hiddenWieghtsLayer[k1].weightN;
                s = s * -1;
                // for l = 2 to l????
                for(int l = 0; l < numInputNodes; l++){
                    double sp1;
                    double sp2;
                    sp1 = hiddenWieghtsLayer[k1].weights[l];
                    //sp1 = hiddenWieghtsLayer[k1].weights[l];
                    sp2 = inputWieghtsLayer[l].actN;
                    s += (sp1 * sp2);    
                }
                hiddenWieghtsLayer[k1].inputN = s;
                hiddenWieghtsLayer[k1].actN = sigmoid(s);
            }

            // Output layer prop
            for(int k2 = 0; k2 < numOutputNodes; k2++){
                s = outputWieghtsLayer[k2].weightN;
                s = s * -1;

                for(int l = 0; l < numHiddenNodes; l++){
                    double sp1;
                    double sp2;
                    sp1 = outputWieghtsLayer[k2].weights[l];
                    //sp1 = hiddenWieghtsLayer[k1].weights[l];
                    sp2 = hiddenWieghtsLayer[l].actN;
                    s += (sp1 * sp2);    
                }
                outputWieghtsLayer[k2].inputN = s;
                outputWieghtsLayer[k2].actN = sigmoid(s);
            }
            // Error Calculations

            // Output layer
            for(int eo = 0; eo < numOutputNodes; eo++){
                double p1;
                double p2;
                double p3;
                p1 = derSigmoid(outputWieghtsLayer[eo].inputN);
                p2 = allExamplesTrain[j][1][eo];
                p3 = outputWieghtsLayer[eo].actN;
                bTrain = p1 * (p2 - p3);
                outErr.push_back(bTrain);
            }

            // Hidden layer
            for(int eh = 0; eh < numHiddenNodes; eh++){
                double newS = 0;
                for(int l = 0; l < numOutputNodes; l++){
                    double sp1;
                    double sp2;
                    sp1 = outputWieghtsLayer[l].weights[eh];
                    sp2 = outErr[l];
                    newS += (sp1 * sp2);
                }
                double sigder;
                sigder = derSigmoid(hiddenWieghtsLayer[eh].inputN);
                bTrain = sigder * newS;
                double p3;
                hidErr.push_back(bTrain);
            }

            // Update every wieght in network using delates

            // Input -> Hidden
            for(int h = 0; h < numHiddenNodes; h++){
                for(int l = 0; l < numInputNodes; l++){
                    double act;
                    act = inputWieghtsLayer[l].actN;
                    double update;

                    update = lr * act * hidErr[h];
                    hiddenWieghtsLayer[h].weights[l] += update; 
                }
                hiddenWieghtsLayer[h].weightN +=  lr * -1 * (hidErr[h]); 
            }

            // Hidden -> output
            for(int o = 0; o < numOutputNodes; o++){
                for(int l = 0; l < numHiddenNodes; l++){
                    double act;
                    act = hiddenWieghtsLayer[l].actN;
                    double update;
                    //MIGHT HAVE TO CHANGE THIS
                    update = lr * act * outErr[o];
                    outputWieghtsLayer[o].weights[l] += update; 
                }
                outputWieghtsLayer[o].weightN +=  lr * -1 * (outErr[o]); 
            }
        }
    }
    //epocs loop ends here

    // Results
    results(outF);
}

void NeuralNet::preTest(){
    cout << "Enter test file: \n";
    cin >> testF;

    cout << "Enter test output file: \n";
    cin >> outTestF;

    getExamples(testF, bTest);

}

void NeuralNet::initializeCM(){
    // Make all metrics 0
    // cm is all the different confunsion matrices
    vector<double> cm = {0, 0, 0, 0};
    for(int i = 0; i < numOutputNodes; i++){
        CM.push_back(cm);
    }
}

void NeuralNet::test(){
    preTest();
    initializeCM();

    for(int j = 0; j < exampleLength; j++){
            
        double s;
        vector<double> hidErr;
        vector<double> outErr;

        // Propagate the inputs foward to compute the outputs
        for(int k = 0; k < numInputNodes; k++){
            inputWieghtsLayer[k].actN = allExamplesTest[j][0][k];
            inputWieghtsLayer[k].inputN = inputWieghtsLayer[k].actN;
        }
        
        // Hidden layer prop
        for(int k1 = 0; k1 < numHiddenNodes; k1++){
            s = hiddenWieghtsLayer[k1].weightN;
            s = s * -1;

            for(int l = 0; l < numInputNodes; l++){
                double sp1;
                double sp2;
                sp1 = hiddenWieghtsLayer[k1].weights[l];
                //sp1 = hiddenWieghtsLayer[k1].weights[l];
                sp2 = inputWieghtsLayer[l].actN;
                s += (sp1 * sp2);    
            }
            hiddenWieghtsLayer[k1].inputN = s;
            hiddenWieghtsLayer[k1].actN = sigmoid(s);
        }

        // Output layer prop
        for(int k2 = 0; k2 < numOutputNodes; k2++){
            s = outputWieghtsLayer[k2].weightN;
            s = s * -1;

            for(int l = 0; l < numHiddenNodes; l++){
                double sp1;
                double sp2;
                sp1 = outputWieghtsLayer[k2].weights[l];
                //sp1 = hiddenWieghtsLayer[k1].weights[l];
                sp2 = hiddenWieghtsLayer[l].actN;
                s += (sp1 * sp2);    
            }
            outputWieghtsLayer[k2].inputN = s;
            outputWieghtsLayer[k2].actN = sigmoid(s);

            int flag;
            flag = allExamplesTest[j][1][k2];

            // Micro-averaging
            if( (sigmoid(s) >= 0.5) && (flag == 1) ){
                CM[k2][0]++;
                globA++;

            } else if( (sigmoid(s) >= 0.5) && (flag == 0) ){
                CM[k2][1]++;
                globB++;

            } else if( (sigmoid(s) < 0.5) && (flag == 1) ){
                CM[k2][2]++;
                globC++;

            } else {
                CM[k2][3]++;
                globD++; 
            }    
        }
    }
    // End of first loop

    calcMetrics(outTestF, CM);
}

void NeuralNet::calcMetrics(string outputFile, vector<vector<double>> CM){

    ofstream of(outputFile);
    
    // Macro Metrics
    double macA;
    double macP;
    double macR;
    double macF;

    // micro Metrics
    double micA;
    double micP;
    double micR;
    double micF;

    double A;
    double B;
    double C;
    double D;

    for(int i = 0; i < numOutputNodes; i++){
        
        A = CM[i][0];
        B = CM[i][1];
        C = CM[i][2];
        D = CM[i][3];

        //accurary = (0 + 3) / (Sum(ALL))
        accurary = (A + D) / (A + B + C + D);
        //precision = 0 / (0 + 1)
        precision = A / (A + B);
        //recall = 0 / (0 + 2)
        recall = A / (A + C);
        
        f1 = (2 * precision * recall) / (precision + recall);

        of << fixed << setprecision(0) << A << " " << B << " " << C << " " << D << " ";
        of << fixed << setprecision(3) << accurary << " " << precision << " " << recall << " " << f1 << endl;

        //cout << i << ": " << A << " " << B << " " << C << " " << D << " " << accurary << " " << precision << " " << recall << " " << f1 << endl;

        macA += accurary;
        macP += precision;
        macR += recall;

    }

    //macros gonna be the average of everything
    macA = macA / numOutputNodes;
    macP = macP / numOutputNodes;
    macR = macR / numOutputNodes;

    macF = (2 * macP * macR) / (macP + macR);


    //cout << "macro stuff: " << macA << " " << macP << " " << macR << " " << macF << endl;

    // Micro results
    micA = (globA + globD) / (globA + globB + globC + globD);
    micP = globA / (globA + globB);
    micR = globA / (globA + globC);

    micF = (2 * micP * micR) / (micP + micR);

    of << fixed << setprecision(3) << micA << " " << micP << " " << micR << " " << micF << endl;

    of << fixed << setprecision(3) << macA << " " << macP << " " << macR << " " << macF << endl;

    //cout << "micro stuff: " << micA << " " << micP << " " << micR << " " << micF << endl;

    of.close();
}