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
    std::vector<std::vector<double>> test_input = minstdata.getTestInput();
    minstdata.printDigit(test_input[0],0);


    /// Model example
    // Create an instance of MyModel
    MyModel model;
    std::cout << "Model initalized!"<< std::endl;

    // Display model details
    model.details();

    // Create a random input vector of size 10
    Eigen::VectorXd input(10);
    input.setRandom();  // Random input for illustration
    std::cout << "Forward passing..."<< std::endl;

    // Perform the forward pass
    Eigen::VectorXd output = model.forward(input);

    // Output the final result
    std::cout << "Final Output: " << output.transpose() << std::endl;

    std::cout << "Backpropagating..."<< std::endl;


    // Create a random gradient (simulating the gradient of the loss with respect to output)
    Eigen::VectorXd dout(10);
    dout.setRandom();  // Random gradient for illustration

    // Perform the backward pass with a learning rate of 0.001
    model.backward(dout);
    std::cout << "End of program"<< std::endl;

    return 0;
}
