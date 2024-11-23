//
// Created by siyan on 2024/10/27.
// Edited by Zhenjiang_Li on 2024/11/23
//
#include "LinearLayer.h"

// Xavier initialization
void LinearLayer::xavier_uniform(Eigen::MatrixXd& weights, int fan_in, int fan_out) {
    double limit = std::sqrt(6.0 / (fan_in + fan_out));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-limit, limit);

    for (int i = 0; i < weights.rows(); ++i) {
        for (int j = 0; j < weights.cols(); ++j) {
            weights(i, j) = dist(gen);
        }
    }
}

// Kaiming initialization
void LinearLayer::kaiming_uniform(Eigen::MatrixXd& weights, int fan_in) {
    double limit = std::sqrt(6.0 / fan_in);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-limit, limit);

    for (int i = 0; i < weights.rows(); ++i) {
        for (int j = 0; j < weights.cols(); ++j) {
            weights(i, j) = dist(gen);
        }
    }
}

// Constant initialization for biases
void LinearLayer::constant(Eigen::VectorXd& bias, double value) {
    bias.setConstant(value);
}

// Constructor
LinearLayer::LinearLayer(int input_size, int output_size, const std::string& init_strategy)
    : input_size_(input_size), output_size_(output_size), 
      weights_(Eigen::MatrixXd::Zero(output_size, input_size)),
      bias_(Eigen::VectorXd::Zero(output_size)),
      input_(Eigen::VectorXd::Zero(input_size)),
      output_(Eigen::VectorXd::Zero(output_size)) {

    if (init_strategy == "xavier_uniform") {
        xavier_uniform(weights_, input_size_, output_size_);
        constant(bias_, 0.1);
    } else if (init_strategy == "kaiming_uniform") {
        kaiming_uniform(weights_, input_size_);
        constant(bias_, 0.1);
    } else {
        throw std::invalid_argument("Unknown initialization strategy");
    }
}

Eigen::VectorXd LinearLayer::forward(const Eigen::VectorXd& input, const std::string & activation) {
    if (input.size() != input_size_) {
        throw std::invalid_argument("Input size must match input_size_");
    }
    input_ = input;  // Save input for backward pass
    Eigen::VectorXd raw_output = (weights_ * input_) + bias_;  // Linear transformation


    if (activation == "relu"){
        output_ = ReLU::relu(raw_output);  // Apply activation function
        return output_;
    }
    else if(activation == "sigmoid"){
        output_ = Sigmoid::sigmoid(raw_output);  // Apply activation function
        return output_;
    }
    else{
        throw std::invalid_argument("Unknown activation")
    }


}


// // Backward propagation (stub, needs implementation)
// Tensor LinearLayer::backward(const Tensor& dout) {
//     //
//     return Tensor();  // Replace with actual implementation
// }
Eigen::MatrixXd LinearLayer::getWeights(){return weights_;}

Eigen::MatrixXd LinearLayer::getBias(){return bias_;}


