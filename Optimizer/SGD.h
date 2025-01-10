#ifndef SGD_H
#define SGD_H

#include <Eigen/Dense>
#include "mymodel.h"

class SGD {
private:
    double learning_rate_;  // 学习率
    MyModel model_;

public:
    // 构造函数
    explicit SGD(MyModel model, double learning_rate) {
        this->model_ = model;
        this->learning_rate_ = learning_rate;
    };
    void step() ;
};

#endif // SGD_H



