//
// Created by siyan on 2024/10/27.
// Edited by Zhenjiang_Li on 2024/11/23
//
#ifndef LINEARLAYER_H
#define LINEARLAYER_H

#include <iostream>
#include <Eigen>
#include <random>
#include <cassert>
#include <cmath>
#include "activation.h"
#include "model.h"
#include "Tensor.h"

class LinearLayer : public Model {
public:
    /// @brief Constructor: initialize linear layer
    /// @param input_size 
    /// @param output_size 
    LinearLayer(int input_size, int output_size, const std::string& init_strategy = "xavier_uniform");

    /// @brief Apply a linear transformation to the input and an optional activation function.
    /// 
    /// This function performs a linear transformation using the weights and biases of the layer:
    ///     output = weights * input + bias
    ///
    /// @param input Input vector of size input_size_. The size of the vector must match the layer's input dimension.
    /// @param activation Supported values are:
    ///                   - "relu": Apply the ReLU activation function.
    ///                   - "sigmoid": Apply the Sigmoid activation function.
    ///
    /// @return Output vector of size output_size_. 
    ///
    /// @throws std::invalid_argument if the input size does not match input_size_
    ///         or if the specified activation function is not supported.
    ///
    /// @note This function also saves the input vector for use during the backward pass (for gradient calculations).
    Eigen::VectorXd forward(const Eigen::VectorXd& input, const std::string & activation = "sigmoid") override; 

    // Tensor backward(const Tensor& dout);

    /// @brief  Get weights' of layer
    /// @return Weights matrix. Type Eigen::MatrixXd
    Eigen::MatrixXd getWeights();

    /// @brief Get bias' of layer
    /// @return Bias vector . Type Eigen::VectorXd
    Eigen::MatrixXd getBias();
    
    /// @brief print details of layers
    void details() const override {
        std::cout << "Linear Layer: Input size = " << input_size_ << ", Output size = " << output_size_ << std::endl;
    }


private:
    int input_size_;              // Input size
    int output_size_;             // Output size
    Eigen::MatrixXd weights_;     // Weights matrix
    Eigen::VectorXd bias_;        // Bias vector
    Eigen::VectorXd input_;       // Store the last input for backward
    Eigen::VectorXd output_;      // Store the last output for backward

    void xavier_uniform(Eigen::MatrixXd& weights, int fan_in, int fan_out);
    void kaiming_uniform(Eigen::MatrixXd& weights, int fan_in);
    void constant(Eigen::VectorXd& bias, double value);
};

#endif // LINEARLAYER_H
