#include "LinearLayer.h"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    // 测试用例
    int input_size = 3;
    int output_size = 2;

    // 创建一个线性层
    LinearLayer linear_layer(input_size, output_size, "kaiming_uniform");

    // 打印初始化的权重和偏置
    std::cout << "Initialized Weights:" << std::endl;
    for (const auto& row : linear_layer.weights_) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Initialized Bias:" << std::endl;
    for (const auto& val : linear_layer.bias_) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // 测试多个输入向量
    std::vector<std::vector<double>> test_inputs = {
        {1.0, 0.5, -1.5},
        {0.0, 1.0, 2.0},
        {-1.0, -0.5, 0.5},
        {0.5, 0.5, 0.5}
    };

    for (size_t i = 0; i < test_inputs.size(); ++i) {
        const auto& input = test_inputs[i];

        // 打印输入向量
        std::cout << "Input Vector " << i + 1 << ": ";
        for (const auto& val : input) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        // 进行前向传播
        std::vector<double> output = linear_layer.forward(input);

        // 打印输出结果
        std::cout << "Output Vector " << i + 1 << ": ";
        for (const auto& val : output) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
