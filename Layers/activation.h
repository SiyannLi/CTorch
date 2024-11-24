//
// Created by ZhenjiangLi on 2024/11/23.
// Edited by Zhenjiang_Li on 2024/11/23
//

#ifndef RELU_H
#define RELU_H

#include <iostream>
#include <Eigen>
#include <cmath>

class ReLU {
public:
    /// @brief Apply ReLU activation function
    /// @param input Input vector (Eigen::VectorXd)
    /// @return Output vector after applying ReLU
    static Eigen::VectorXd relu(const Eigen::VectorXd& input) {
        Eigen::VectorXd result = input;
        for (int i = 0; i < result.size(); ++i) {
            result[i] = std::max(0.0, result[i]);
        }
        return result;
    }

    /// @brief Compute the gradient of ReLU activation
    /// @param input Input vector (Eigen::VectorXd)
    /// @return Gradient vector of the same size
    static Eigen::VectorXd gradient(const Eigen::VectorXd& input) {
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(input.size());
        for (int i = 0; i < input.size(); ++i) {
            grad[i] = (input[i] > 0) ? 1.0 : 0.0;
        }
        return grad;
    }
};

class Sigmoid {
public:
    /// @brief Apply Sigmoid activation function
    /// @param input Input vector (Eigen::VectorXd)
    /// @return Output vector after applying Sigmoid
    static Eigen::VectorXd sigmoid(const Eigen::VectorXd& input) {
        Eigen::VectorXd result = input;
        for (int i = 0; i < result.size(); ++i) {
            result[i] = 1.0 / (1.0 + std::exp(-result[i]));
        }
        return result;
    }

    /// @brief Compute the gradient of Sigmoid activation
    /// @param input Input vector (Eigen::VectorXd)
    /// @return Gradient vector of the same size
    static Eigen::VectorXd gradient(const Eigen::VectorXd& input) {
        Eigen::VectorXd sig = sigmoid(input);
        return sig.array() * (1.0 - sig.array());
    }
};

#endif // RELU_H
