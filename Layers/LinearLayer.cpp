//
// Created by siyan on 2024/10/27.
//

#include "LinearLayer.h"

void Init::xavier_uniform(std::vector<std::vector<double>>& weights, int fan_int, int fan_out)
{
    double limit = std::sqrt(6.0 / (fan_int + fan_out ));
}

void Init::kaiming_uniform(std::vector<std::vector<double>>& eights, int fan_in)
{

}

void Init::constant(std::vector<double>& bias, double value)
{

}

void Init::initialize_uniform(std::vector<std::vector<double>>& matrix, double min, double max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min, max);  


}

LinearLayer::LinearLayer(int input_size, int out_size):
input_size_(input_size), output_size_(out_size)
{

}

std::vector<double> LinearLayer::forward(const std::vector<double>& input)
{

}