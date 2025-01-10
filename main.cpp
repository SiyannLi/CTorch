#include <iostream>
#include "Data/DataLoader.h"
#include "DataSet.h"
#include "LinearLayer.h"
#include "Eigen/Dense"
#include "mymodel.h"
#include "Optimizer/SGD.h"

int main() {
    // Load dataset
    std::cout << "Loading date..." << std::endl;
    DataSet minstdata;
    minstdata.readMnistData();
    Eigen::MatrixXd train_input = minstdata.getTrainInput();
    Eigen::MatrixXd train_output = minstdata.getTrainOutput();
    std::cout << "Load correspoing image for training:" << std::endl;
    minstdata.printDigit(train_input.row(2), 0);
    std::cout << "Load correspoing image label(one-hot encoded):" << std::endl;
    std::cout << train_output.row(2) << std::endl;
    std::cout << "---------------------------------------------" << std::endl;

    // Create model instance
    MyModel model;
    double learning_rate = 0.0001;

    // Create optimizer instance, pass model pointer
    SGD optimizer(&model, learning_rate);
    std::cout << "Model initialized!" << std::endl;

    // Display model details
    model.details();
    std::cout << "---------------------------------------------" << std::endl;
    Eigen::VectorXd input = train_input.row(2);
    std::cout << "Input size: " << input.size() << std::endl;
    std::cout << "Forward passing..." << std::endl;

    // Forward pass
    Eigen::VectorXd output = model.forward(input);
    Eigen::MatrixXd output_new = output.transpose();
    std::cout << "---------------------------------------------" << std::endl;

    // Compute loss
    Eigen::VectorXd labels = train_output.row(0);
    Eigen::VectorXi labels_new = labels.cast<int>();
    std::cout << "Loss computing..." << std::endl;

    CrossEntropy loss(output_new, labels_new);
    double loss_output = loss.forward();
    std::cout << "Cross-Entropy Loss: " << loss_output << std::endl;
    std::cout << "---------------------------------------------" << std::endl;

    // Backpropagation
    std::cout << "Backpropagating..." << std::endl;
    Eigen::VectorXd dout = loss.backward().row(0);
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "Gradient (dout): " << std::endl << dout.transpose() << std::endl;
    std::cout << "---------------------------------------------" << std::endl;

    Eigen::VectorXd grad = model.backward(dout, "sigmoid");
    std::cout << "Gradient (backward pass): " << grad.transpose() << std::endl;


    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "Performing SGD..." << std::endl;
    // Update parameters
    optimizer.step();
    std::cout << "Parameters updated!" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;

    std::cout << "End of program!!" << std::endl;
    return 0;
}
