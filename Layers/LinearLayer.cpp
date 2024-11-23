    //
    // Created by siyan on 2024/10/27.
    //

    #include "LinearLayer.h"

    void LinearLayer::initialize_uniform(std::vector<std::vector<double>>& matrix, double min, double max)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(min, max);  
        
        for (auto& row:matrix){
            for(auto& val: row){
                val = dist(gen);
            }
        }

    }

    void LinearLayer::xavier_uniform(std::vector<std::vector<double>>& weights, int fan_int, int fan_out)
    {
        double limit = std::sqrt(6.0 / (fan_int + fan_out ));
        initialize_uniform(weights, -limit, limit);
    }

    void LinearLayer::kaiming_uniform(std::vector<std::vector<double>>& weights, int fan_in)
    {
        double limit = std::sqrt(6.0 / (fan_in));
        initialize_uniform(weights, -limit, limit);

    }

    void LinearLayer::constant(std::vector<double>& bias, double value)
    {
        std::fill(bias.begin(), bias.end(), value);
    }



    LinearLayer::LinearLayer(int input_size, int output_size, 
        const std::string& init_strategy):
        input_size_(input_size), output_size_(output_size), 
        weights_(output_size, std::vector<double>(input_size, 0.0)),
        bias_(output_size, 0.0) 
    {
        if (init_strategy == "xavier_uniform") {
            xavier_uniform(weights_, input_size_, output_size_);
            constant(bias_, 0.1); 
        } else if (init_strategy == "kaiming_uniform") {
            kaiming_uniform(weights_, input_size_);
            constant(bias_, 0.1); 
        } else {
            throw std::invalid_argument("Unknown initialization strategy");
        }
    }

    Tensor LinearLayer::forward(const std::vector<double>& input)
    {
        if (input.size() != input_size_) {
            throw std::invalid_argument("Input size must match input_size_");
        }
        std::vector<double> output(output_size_, 0); 
        for (size_t i = 0; i < weights_.size(); ++i) {
            for (size_t j = 0; j < weights_[0].size(); ++j) {
                output[i] += weights_[i][j] * input[j];
            }
            output[i] += bias_[i]; // Combine bias addition
        }
        return output;
    }




