#include "CrossEntropy.h"

// 构造函数
CrossEntropy::CrossEntropy(const MatrixXd& logits, const Eigen::VectorXi& labels)
    : logits(logits), labels(labels) {
    softmax = computeSoftmax(logits);
}

// 前向传播
double CrossEntropy::forward() {
    double loss = 0.0;
    for (int i = 0; i < logits.rows(); ++i) {
        loss += -std::log(softmax(i, labels[i]));  // 选取每个样本对应类别的 softmax 概率
    }
    return loss / logits.rows();  // 平均损失
}

// 反向传播
MatrixXd CrossEntropy::backward() {
    MatrixXd grad = softmax;
    for (int i = 0; i < logits.rows(); ++i) {
        grad(i, labels[i]) -= 1;  // 对应类别位置减 1
    }
    return grad / logits.rows();  // 平均梯度
}

// 计算 Softmax
MatrixXd CrossEntropy::computeSoftmax(const MatrixXd& input) const {
    MatrixXd max_vals = input.rowwise().maxCoeff().replicate(1, input.cols());
    MatrixXd exp_scores = (input.array() - max_vals.array()).exp();
    return exp_scores.array().colwise() / exp_scores.rowwise().sum().array();
}
