//
// Created by zhenjiang-li on 1/10/25.
//

#ifndef NETWORK_H
#define NETWORK_H

#include <Eigen/Dense>
#include  "LinearLayer.h"
#include  "model.h"
#include <memory>



class Network {
public:
    Network(const Eigen::VectorXd& input, const Eigen::VectorXd& outputs, const Eigen:: VectorXd& net_layer,const std::string& activation, const std::string& optimizer) {
        Eigen::VectorXd full_net(net_layer.size() + 2);
        full_net[0] = input.size();
        full_net.segment(1, net_layer.size()) = net_layer;
        full_net(full_net.size() - 1) = outputs.size();

        for (int i = 0; i < full_net.size() - 1; i++) {
            int inputsize = full_net[i];
            int outputsize = full_net[i + 1];
            layers_.emplace_back((std::make_unique<LinearLayer>(inputsize, outputsize)));
        }
    };


private:
    std::vector<std::unique_ptr<Model>> layers_;
    std::vector<Eigen::VectorXd> weights_;

    Eigen::VectorXd forward(const Eigen::VectorXd& input, const std::string& activation ="signoid") {
        Eigen::VectorXd x = input;
        for(const auto& layer : layers_){
            x = layer->forward(x, "sigmoid");
        }
        std::cout << "Output: " << x.transpose() << std::endl;
        return x;
    }
    Eigen::VectorXd backward(const Eigen::VectorXd& dout, const std::string& activation = "sigmoid", double learning_rate = 0.0001)  {
        Eigen::VectorXd grad = dout;  // init grad inhereted from last layer
        // Prints the details of the model and its layers.
        // This function is primarily used for debugging and model introspection.
        for (int i = layers_.size() - 1; i >= 0; --i) {
            grad = layers_[i]->backward(grad, activation, learning_rate);  // back propagate of each layer

        std::cout << "Gradients of last layer: " << grad.transpose() << std::endl;
        return grad;  // return the gradient of last layer
    }




};

#endif //NETWORK_H
