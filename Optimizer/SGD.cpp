#include "SGD.h"
#include <stdexcept>

// 构造函数
SGD::SGD(MyModel* model, double learning_rate)
    : model_(model), learning_rate_(learning_rate) {}

// step 方法定义
void SGD::step() {
    if (model_) {
        model_->update(learning_rate_);
    } else {
        throw std::runtime_error("Model is not initialized.");
    }
}
