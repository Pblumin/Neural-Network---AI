#include <vector>
#include <string>
#include <iostream>
#include <ctime>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "NN.h"
using namespace std;

int main(){

    string inputFile;
    string outputFile;
    int t;

    //cout<< "Hello world\n";
    cout << "Enter 0 for Train and 1 for Test:\n";
    cin >> t;
    
    while( (t != 0) && (t != 1) ){
        cout << "Please enter 0 or 1.\n";
        cin >> t;
    }

    if(t == 0){
        cout << "You selected train. Please enter a valid input file.\n";
        cin >> inputFile;
        //NeuralNet nn(inputFile);
        NeuralNet nn = NeuralNet(inputFile);
        nn.flag = 0;
        nn.train();
    } else {
        cout << "You selected test. Please enter a valid input file.\n";
        cin >> inputFile;
        //NeuralNet nn(inputFile);
        NeuralNet nn = NeuralNet(inputFile);
        nn.flag = 1;
        nn.test();
    }

    return 0;

}
