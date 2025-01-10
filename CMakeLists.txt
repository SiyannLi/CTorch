cmake_minimum_required(VERSION 3.30)
project(CTorch)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(Data)
include_directories(DataType)
include_directories(Eigen)
include_directories(Function)
include_directories(Layers)
include_directories(Optimizer)

add_executable(CTorch
    Data/DataLoader.cpp
    Data/DataLoader.h
    Data/DataSet.cpp
    Data/DataSet.h
    Function/LossFunction.h
    Layers/activation.cpp
    Layers/activation.h
    Layers/ConvolutionLayer.cpp
    Layers/ConvolutionLayer.h
    Layers/LinearLayer.cpp
    Layers/LinearLayer.h
    Layers/model.cpp
    Layers/model.h
    Layers/Module.h
    Optimizer/Adam.cpp
    Optimizer/Adam.h
    Optimizer/Optimizer.h
    main.cpp
    mymodel.h)
