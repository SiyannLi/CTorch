cmake_minimum_required(VERSION 3.29)
project(CTorch)

set(CMAKE_CXX_STANDARD 20)

add_executable(CTorch main.cpp
        DataType/Tensor.cpp
        Layers/LinearLayer.cpp
        Layers/LinearLayer.h
        Layers/ConvolutionLayer.cpp
        Layers/ConvolutionLayer.h
        Optimizer/Adam.cpp
        Optimizer/Adam.h
        DataType/Tensor.cpp
        DataType/Tensor.h
        Data/DataLoader.cpp
        Data/DataLoader.h
        Data/DataSet.cpp
        Data/DataSet.h
        Layers/ReLU.cpp
        Layers/ReLU.h)
