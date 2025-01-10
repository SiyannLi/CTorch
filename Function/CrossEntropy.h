#ifndef CROSSENTROPY_H
#define CROSSENTROPY_H

#include <Eigen/Dense>
#include <iostream>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXi;

class CrossEntropy {
private:
    MatrixXd logits;   // 输入的未归一化得分
    VectorXi labels;   // 实际的类别索引
    MatrixXd softmax;  // Softmax 的输出值

    // 计算 softmax
    MatrixXd computeSoftmax(const MatrixXd& input) const;

public:
    // 构造函数
    CrossEntropy(const MatrixXd& logits, const VectorXi& labels);

    // 计算损失值
    double forward();

    // 计算梯度
    MatrixXd backward();
};

#endif // CROSSENTROPY_H
