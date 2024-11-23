<<<<<<< HEAD
//
// Created by siyan on 2024/10/27.
//

#include "LinearLayer.h"

void Init::xavier_uniform(std::vector<std::vector<double>>& weights, int fan_int, int fan_out)
{
    double limit = std::sqrt(6.0 / (fan_int + fan_out ));
    initialize_uniform(weights, -limit, limit);
}

void Init::kaiming_uniform(std::vector<std::vector<double>>& weights, int fan_in)
{
    double limit = std::sqrt(6.0 / (fan_in));
    initialize_uniform(weights, -limit, limit);float

}

void Init::constant(std::vector<double>& bias, double value)
{
    std::fill(bias.begin(), bias.end(), value);
}

void Init::initialize_uniform(std::vector<std::vector<double>>& matrix, double min, double max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min, max);  
    
    for (auto& row:matrix){
        for(auto& val: row){
            val = dist(gen);
        }
    }

}

LinearLayer::LinearLayer(int input_size, int output_size):
input_size_(input_size), output_size_(output_size)
{
    std::vector<std::vector<double>> weights_(output_size, std::vector<double>(input_size, 0.0));
    std::vector<double> bias_(output_size, 0.0);
}

std::vector<double> LinearLayer::forward(const std::vector<double>& input)
{
    assert(input.size() == input_size_ && "Input must match input size!");
    std::vector<double> output(output_size_); 
    output = weights_ * input + bias_;

}
=======
//
// Created by siyan on 2024/10/27.
//

#include "LinearLayer.h"
>>>>>>> origin/siyan
