//
// Created by Zhenjiang_Li on 2024/11/23.
// Edited by Zhenjiang_Li on 2024/11/23
//

#ifndef MYMODEL_H
#define MYMODEL_H

#include <memory>
#include <vector>
#include "model.h"
#include "LinearLayer.h"

class MyModel {
public:
    MyModel() {
        layers_.emplace_back(std::make_unique<LinearLayer>(10,50));
        layers_.emplace_back(std::make_unique<LinearLayer>(50,30));
        layers_.emplace_back(std::make_unique<LinearLayer>(30,10));
    }

    Eigen::VectorXd forward(const Eigen::VectorXd& input, const std::string& activation = "")  {
        Eigen::VectorXd x = input;
        for(const auto& layer : layers_){
            x = layer->forward(x, "sigmoid");
        }
        std::cout << "Output: " << x.transpose() << std::endl;
        return x;
    }
    void details() const  {
        std::cout << "MyModel:" << std::endl;
        for (const auto& layer : layers_) {
            layer->details();
        }
    }

private:
    std::vector<std::unique_ptr<Model>> layers_;
};




#endif // MYMODEL_H