#include "LinearLayer.h"
#include <iostream>
#include <Eigen>

#include "mymodel.h"
#include <Eigen/Dense>

int main() {

    MyModel model;


    model.details();


    Eigen::VectorXd input(10);
    input.setRandom();


    Eigen::VectorXd output = model.forward(input);


    std::cout << "Final Output: " << output.transpose() << std::endl;

    return 0;
}

/*
void debug_output(const Eigen::VectorXd& input, LinearLayer& layer) {
    // 打印输入
    std::cout << "Input:\n" << input << std::endl;

    // 前向传播
    Eigen::VectorXd output = layer.forward(input);

    // 打印输出
    std::cout << "Output:\n" << output << std::endl;
}

int main() {
    int input_size = 3;
    int output_size = 2;

    // 创建线性层，使用 Xavier 初始化
    LinearLayer layer(input_size, output_size, "xavier_uniform");

    // 打印权重和偏置
    std::cout << "Initialized Weights:\n" << layer.getWeights() << std::endl;
    std::cout << "Initialized Bias:\n" << layer.getBias() << std::endl;

    // 定义多个输入
    std::vector<Eigen::VectorXd> inputs = {
        (Eigen::VectorXd(3) << 1.0, 2.0, 3.0).finished(),
        (Eigen::VectorXd(3) << 0.5, -1.0, 0.0).finished(),
        (Eigen::VectorXd(3) << -1.0, 0.5, 2.0).finished()
    };

    // 遍历输入并进行前向传播
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::cout << "Case " << i + 1 << ":" << std::endl;
        debug_output(inputs[i], layer);
        std::cout << "-------------------" << std::endl;
    }

    return 0;
}


*/
