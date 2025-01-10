#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>

// Base class for a layer
class Model {
public:
    virtual ~Model() = default;
    virtual void printInfo() const = 0;
};

// Derived class for a linear layer
class LinearLayer : public Model {
private:
    int inputSize_;
    int outputSize_;
public:
    LinearLayer(int inputSize, int outputSize)
        : inputSize_(inputSize), outputSize_(outputSize) {}

    void printInfo() const override {
        std::cout << "Linear Layer: " << inputSize_ << " -> " << outputSize_ << std::endl;
    }
};

int main() {
    // Define the original net_layer without input/output sizes
    Eigen::VectorXd net_layer(2);
    net_layer << 50, 30; // Hidden layers only

    // Define input and output sizes
    int inputSize = 10;  // Example input size
    int outputSize = 10; // Example output size

    // Add input and output sizes to net_layer
    Eigen::VectorXd updated_net_layer(net_layer.size() + 2);
    updated_net_layer[0] = inputSize; // First layer (input)
    updated_net_layer.tail(net_layer.size()) = net_layer; // Hidden layers
    updated_net_layer[updated_net_layer.size() - 1] = outputSize; // Last layer (output)

    // Print the updated network structure
    std::cout << "Updated net_layer: " << updated_net_layer.transpose() << std::endl;

    // Vector to hold layers
    std::vector<std::unique_ptr<Model>> layers_;

    // Construct layers based on updated_net_layer
    for (int i = 0; i < updated_net_layer.size() - 1; ++i) {
        int inputSize = updated_net_layer[i];
        int outputSize = updated_net_layer[i + 1];
        layers_.emplace_back(std::make_unique<LinearLayer>(inputSize, outputSize));
    }

    // Print layer information
    std::cout << "Network structure:" << std::endl;
    for (const auto& layer : layers_) {
        layer->printInfo();
    }

    return 0;
}
