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
    #include "DataType/Tensor.h"


    class LinearLayer {
    public:
        std::vector<std::vector<double>> weights_;  // weights matrix
        std::vector<double> bias_;                   // bias vector


        /// @brief Constructor: initialize linear layer
        /// @param input_size 
        /// @param output_size 
        LinearLayer(int input_size, int output_size, const std::string& init_strategy = "xavier_uniform");
        

        /// @brief Apply a linear transformation to the input.
        /// @param input Input vector of size input_size_.
        /// @return Output vector of size output_size_.
        std::vector<double> forward(const std::vector<double>& input);

        static void xavier_uniform(std::vector<std::vector<double>>& weights, int fan_int, int fan_out);
        static void kaiming_uniform(std::vector<std::vector<double>>& weights, int fan_in);
        static void constant(std::vector<double>& bias, double value);

    private:
        int input_size_;                            // input size
        int output_size_;                           // output size


        static void initialize_uniform(std::vector<std::vector<double>>& matrix, double min, double max);


    };



    #endif //LINEARLAYER_H
