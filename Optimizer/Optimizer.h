//
// Created by Siyan Li on 11/23/24.
//

#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <stdexcept>

#include "Module.h"
#include "LossFunction.h"
class Optimizer {
private:


public:
    double learningRate;
    Module model;
    LossFunction loss;

    Optimizer(Module model, LossFunction loss, double learningRate ) {
        this ->loss = loss;
        this -> model = model;
        this -> learningRate = learningRate;
    }

    void zeroGrad() {
        throw std::logic_error("This function is not yet implemented.");
    }

    void step() {
        throw std::logic_error("This function is not yet implemented.");
    }
};

#endif //OPTIMIZER_H
