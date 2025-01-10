//
// Created by Zhenjiang_Li on 2024/11/23.
// Edited by Zhenjiang_Li on 2024/11/23
//
#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <Eigen>

class Model{
    public:

    /// @brief virtual funtion
    /// @param input 
    /// @param activation 
    /// @return 
    virtual Eigen::VectorXd forward(const Eigen::VectorXd& input, const std::string& activation = "") = 0;

    
    virtual Eigen::VectorXd backward(const Eigen::VectorXd& dout, const std::string& activation = "") = 0;


    virtual void details() const{
        std::cout << "Generic model." <<std::endl; 
    }
    
    
    virtual ~Model() = default;
};



#endif // MODEL_H