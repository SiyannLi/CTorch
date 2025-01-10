#include <iostream>
#include "Data/DataLoader.h"
#include "Data/DataSet.h"
#include "LinearLayer.h"
#include <iostream>
#include <Eigen>
#include "mymodel.h"
// #include <Eigen/Dense>

int main() {
    /// load datatset
    DataSet minstdata;
    minstdata.readMnistData();
    Eigen::MatrixXd train_input = minstdata.getTrainInput();
    Eigen::MatrixXd train_output = minstdata.getTrainOutput();
    minstdata.printDigit(train_input.row(2),0);
    std::cout<<train_output.row(2)<<std::endl;


    // Model example
    // Create an instance of MyModel
    MyModel model;
    std::cout << "Model initalized!"<< std::endl;

    // Display model details
    model.details();

    // Create a random input vector of size 10
    // Eigen::VectorXd input(10);
    // input.setRandom();  // Random input for illustration

    Eigen::VectorXd input = train_input.row(2);
    std::cout << "Input size"<< input.size()<< std::endl;
    std::cout << "Forward passing..."<< std::endl;

    // Perform the forward pass
    Eigen::VectorXd output = model.forward(input);

    // Output the final result
    std::cout << "Final Output: " << output.transpose() << std::endl;


    Eigen::VectorXd labels = train_output.row(0);
    std::cout << "Loss computing..."<< std::endl;

    CrossEntropy loss(output, labels);
    double loss_output = loss.forward();
    std::cout << "CE　loss is :"<<loss_output << std::endl;

    std::cout << "Backpropagating..."<< std::endl;

    Eigen::VectorXd dout = loss.backward();
    std::cout << "CE dout ="<< dout << std::endl;
    // Create a random gradient (simulating the gradient of the loss with respect to output)
    // Eigen::VectorXd dout(10);
    // dout.setRandom();  // Random gradient for illustration

    // Perform the backward pass with a learning rate of 0.001
    //
    Eigen::VectorXd grad = model.backward(dout,"sigmoid");
    std::cout << grad.transpose() << std::endl;
    std::cout << "End of program"<< std::endl;

    // Eigen::VectorXd logits(10);
    // logits << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
    //
    // Eigen::VectorXd labels(10);
    // labels << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    //
    // CrossEntropy cross_entropy(logits, labels);
    // std::cout << "---------------------------------------------" << std::endl;
    // double loss = cross_entropy.forward();
    // std::cout << loss << std::endl;
    // Eigen::VectorXd grad = cross_entropy.backward();
    // std::cout << grad << std::endl;
    // std::cout << "---------------------------------------------" << std::endl;

    return 0;
}
