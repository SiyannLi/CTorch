//
// Created by Zhenjiang_Li on 2024/11/23.
// Edited by Zhenjiang_Li on 2024/11/23
//

#ifndef MYMODEL_2_H
#define MYMODEL_2_H

#include <vector>
#include "model.h"
#include <memory>
#include "LinearLayer.h"
#include "Function/CrossEntropy.h"

// MyModel is a class that implements a multi-layer model.
// It inherits from the base Model class and contains linear layers with specific dimensions.
class MyModel : public Model{
public:
    // Constructor: Initializes the model with three linear layers.
    // The dimensions of the layers are as follows:
    // - Input: 10, Output: 50
    // - Input: 50, Output: 30
    // - Input: 30, Output: 10
    MyModel() {
        layers_.emplace_back(std::make_unique<LinearLayer>(784,50));
        layers_.emplace_back(std::make_unique<LinearLayer>(50,30));
        layers_.emplace_back(std::make_unique<LinearLayer>(30,10));
    }
    // Forward pass of the model.
    // Applies each layer in sequence with the specified activation function.
    // @param input: Input vector to the model.
    // @param activation: Activation function to apply at each layer.
    // @return: Output vector after passing through all layers.
    Eigen::VectorXd forward(const Eigen::VectorXd& input, const std::string& activation ="signoid") override {
        Eigen::VectorXd x = input;
        for(const auto& layer : layers_){
            x = layer->forward(x, "sigmoid");
        }

        std::cout << "Output: " << x.transpose() << std::endl;
        return x;
    }

    // Backward pass of the model.
    // Computes gradients and updates weights for all layers in reverse order.
    // @param dout: Gradient of the loss with respect to the output.
    // @param activation: Activation function used in the forward pass .
    // @param learning_rate: Learning rate for updating weights (default: 0.0001).
    // @return: Gradient of the loss with respect to the input.
    Eigen::VectorXd backward(const Eigen::VectorXd& dout, const std::string& activation = "sigmoid") override {
        Eigen::VectorXd grad = dout;  // init grad inhereted from last layer
        // Prints the details of the model and its layers.
        // This function is primarily used for debugging and model introspection.
        for (int i = layers_.size() - 1; i >= 0; --i) {
            grad = layers_[i]->backward(grad, activation);  // back propagate of each layer
        }

        std::cout << "Gradients of last layer: " << grad.transpose() << std::endl;
        return grad;  // return the gradient of last layer

    }

    Eigen::VectorXd update(double learning_rate)  {
        Eigen::VectorXd grad;  // init grad inhereted from last layer
        // Prints the details of the model and its layers.
        // This function is primarily used for debugging and model introspection.
        for (int i = layers_.size() - 1; i >= 0; --i) {
            grad = layers_[i]->update(learning_rate);  // back propagate of each layer
        }

        std::cout << "Gradients of last layer: " << grad.transpose() << std::endl;
        return grad;  // return the gradient of last layer

    }

    void details() const override {
        std::cout << "MyModel:" << std::endl;
        for (const auto& layer : layers_) {
            layer->details();
        }
    }

private:
    // A vector of unique pointers to the layers in the model.
    // Each layer is a unique instance of the LinearLayer class (or derived classes).
    std::vector<std::unique_ptr<Model>> layers_;
};




#endif // MYMODEL_2_H