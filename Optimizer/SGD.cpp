#include "SGD.h"

// 更新参数
void SGD::step() {
    model.update(this->learning_rate_);
}