
//
// Created by siyan on 2024/10/27.
//

#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <fstream>
#include <string>
#include <Eigen>

class DataSet {
private:
    Eigen::MatrixXd train_input;
    Eigen::MatrixXd train_output;
    Eigen::MatrixXd test_input;
    Eigen::MatrixXd test_output;
    std::vector<std::string> label;

    void readMnistTrainImage();
    void readMnistTrainLable();
    void readMnistTestImage();
    void readMnistTestLable();

public:
    DataSet();

    void readMnistData();

    void printDigit(Eigen::VectorXd x, double mask);

    Eigen::MatrixXd getNormalizedData(Eigen::MatrixXd);

    Eigen::MatrixXd getInput();
    Eigen::MatrixXd getOutput();

    Eigen::MatrixXd getTestInput();
    Eigen::MatrixXd getTestOutput();

    ~DataSet();
};



#endif //DATASET_H
