#ifndef SGD_H
#define SGD_H

#include <Eigen/Dense>
#include "mymodel.h"

class SGD {
private:
    double learning_rate_;  // 学习率
    MyModel* model_;        // 使用指针

public:
    // 构造函数
    explicit SGD(MyModel* model, double learning_rate);

    // 更新模型的参数
    void step();
};

#endif // SGD_H
