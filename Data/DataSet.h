
//
// Created by siyan on 2024/10/27.
//

#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <fstream>
#include <string>

class DataSet {
private:
    std::vector<std::vector<double>> train_input;
    std::vector<std::vector<double>> train_output;
    std::vector<std::vector<double>> test_input;
    std::vector<std::vector<double>> test_output;
    std::vector<std::string> label;

    void readMnistTrainImage();
    void readMnistTrainLable();
    void readMnistTestImage();
    void readMnistTestLable();

public:
    DataSet();

    void readMnistData();

    void printDigit(std::vector<double>, double mask);

    std::vector<std::vector<double>> getNormalizedData(std::vector<std::vector<double>>);

    std::vector<std::vector<double>> getInput();
    std::vector<std::vector<double>> getOutput();

    std::vector<std::vector<double>> getTestInput();
    std::vector<std::vector<double>> getTestOutput();

    ~DataSet();
};



#endif //DATASET_H
