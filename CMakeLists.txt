cmake_minimum_required(VERSION 3.30)
project(CTorch)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(include)
include_directories(Eigen)
include_directories(Data)
include_directories(Function)
include_directories(Layers)
include_directories(Network)
include_directories(Optimizer)

add_executable(CTorch
        Data/DataLoader.cpp
        Data/DataLoader.h
        Data/DataSet.cpp
        Data/DataSet.h
        Function/CrossEntropy.cpp
        Function/CrossEntropy.h
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
        Network/Network.h
        Optimizer/Adam.cpp
        Optimizer/Adam.h
        Optimizer/Optimizer.h
    main.cpp
    mymodel.h)
