cmake_minimum_required(VERSION 3.28)
project(CTorch)

set(CMAKE_CXX_STANDARD 20)

include_directories(
        # ${CMAKE_SOURCE_DIR}/DataType
        # ${CMAKE_SOURCE_DIR}/Data
        ${CMAKE_SOURCE_DIR}/Optimizer
        ${CMAKE_SOURCE_DIR}/Layers
       ${CMAKE_SOURCE_DIR}/Eigen
        ${CMAKE_SOURCE_DIR}/
)

add_executable(CTorch main.cpp
        # DataType/Tensor.cpp
        Layers/LinearLayer.cpp
        Layers/LinearLayer.h
        Layers/ConvolutionLayer.cpp
        Layers/ConvolutionLayer.h
        Layers/model.h
        Layers/model.cpp
        Optimizer/Adam.cpp
        Optimizer/Adam.h
        # DataType/Tensor.cpp
        # DataType/Tensor.h
        Data/DataLoader.cpp
        Data/DataLoader.h
        Data/DataSet.cpp
        Data/DataSet.h
        mymodel.h
        Optimizer/Optimizer.h
        Layers/Module.h
        Function/LossFunction.h
        Function/CrossEntropy.cpp
        Function/CrossEntropy.h
        Network/Network.h)
