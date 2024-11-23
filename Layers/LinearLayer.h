<<<<<<< HEAD
//
// Created by siyan on 2024/10/27.
//
#ifndef LINEARLAYER_H
#define LINEARLAYER_H

#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>


class Init{
public:
    static void xavier_uniform(std::vector<std::vector<double>>& weights, int fan_int, int fan_out);
    static void kaiming_uniform(std::vector<std::vector<double>>& eights, int fan_in);
    static void constant(std::vector<double>& bias, double value);
private:
    static void initialize_uniform(std::vector<std::vector<double>>& matrix, double min, double max);
};



class LinearLayer {
public:
    /// @brief Constructor: initialize linear layer
    /// @param input_size 
    /// @param output_size 
    LinearLayer(int input_size, int output_size);

    /// @brief 
    /// @param input 
    /// @return 
    std::vector<double> forward(const std::vector<double>& input);






private:
    int input_size_;                            // input size
    int output_size_;                           // output size
    std::vector<std::vector<double>> weights_;  // weights matrix
    std::vector<float> bias_;                   // bias vector

};



#endif //LINEARLAYER_H
=======
//
// Created by siyan on 2024/10/27.
//

#ifndef LINEARLAYER_H
#define LINEARLAYER_H



class LinearLayer {

};



#endif //LINEARLAYER_H
>>>>>>> origin/siyan
