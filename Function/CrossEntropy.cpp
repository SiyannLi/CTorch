#include "CrossEntropy.h"

// 计算 softmax 函数
MatrixXd CrossEntropy::computeSoftmax(const MatrixXd& input) const {
    // 计算每行最大值并扩展为矩阵
    MatrixXd max_vals = input.rowwise().maxCoeff().replicate(1, input.cols());

    // 使用数组操作进行逐元素计算
    MatrixXd exp_scores = (input - max_vals).array().exp();

    // 归一化 softmax 结果
    return exp_scores.array().colwise() / exp_scores.rowwise().sum().array();
}


// 构造函数
CrossEntropy::CrossEntropy(const MatrixXd& logits, const MatrixXd& labels)
    : logits(logits), labels(labels) {
    // 初始化 softmax
    softmax = computeSoftmax(logits);
}

// 前向传播，计算交叉熵损失
double CrossEntropy::forward() {
    MatrixXd log_probs = (softmax.array().log() * labels.array());
    return -log_probs.sum() / logits.rows();  // 平均损失
}

// 反向传播，计算梯度
MatrixXd CrossEntropy::backward() {
    MatrixXd grad = (softmax - labels) / logits.rows();  // 每个样本的平均梯度
    return grad;
}
