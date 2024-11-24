//
// Created by siyan on 2024/10/27.
//

#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
#include "Eigen"
using namespace std;
template<typename T>
class Tensor {
private:

    vector<T> data;
    vector<T> grad;
    vector<int> dimensions;

    size_t rows() const { return data.size(); }
    size_t cols() const { return data.empty() ? 0 : data[0].size(); }

    vector<T> constructRecursive(const vector<int>& dims, size_t level = 0) {
        if (level == dims.size() - 1) {
            return vector<T>(dims[level], 0);
        }

        vector<T> result(dims[level]);
        for (size_t i = 0; i < dims[level]; ++i) {
            result[i] = constructRecursive(dims, level + 1);
        }
        return result;
    }


public:

    Tensor(const vector<vector<T>>& rawData) {
        data = rawData;
        grad = vector<vector<T>>(data.size(), vector<T>(data[0].size(), 0));
    void getDimensions(const T& vec, vector<size_t>& dims) {
        dims.push_back(vec.size());
        if (!vec.empty() && vec[0].size() > 0) {
            getDimensions(vec[0], dims);
        }
    }


public:
    Tensor(const vector<T>& rawData) {
        this -> data = rawData;
        this -> dimensions = getDimensions(rawData);
        this -> grad = constructRecursive(this -> dimensions, 0);
    }

    Tensor operator+(const Tensor& other) const {

    }

    Tensor operator-(const Tensor& other) const {

    }

    Tensor operator*(const Tensor& other) const {

    }


    
    vector<vector<T>> data;
    vector<vector<T>> grad;
    vector<int> dimensions;

};



#endif //TENSOR_H
