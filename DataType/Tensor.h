//
// Created by siyan on 2024/10/27.
//

#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
using namespace std;
template<typename T>
class Tensor {
private:


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
    }


    Tensor operator+(const Tensor& other) const {
        if (rows() != other.rows() || cols() != other.cols()) {
            throw invalid_argument("Dimension mismatch for addition");
        }

        vector<vector<T>> result(rows(), vector<T>(cols()));
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return Tensor(result);
    }

    Tensor operator-(const Tensor& other) const {
        if (!sameShape(this->data, other.data)) {
            throw invalid_argument("Dimension mismatch for subtraction");
        }

        vector<vector<T>> result(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            result[i].resize(data[i].size());
            for (size_t j = 0; j < data[i].size(); ++j) {
                result[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return Tensor(result);
    }

    Tensor operator*(const Tensor& other) const {
        if (!sameShape(this->data, other.data)) {
            throw invalid_argument("Dimension mismatch for multiplication");
        }

        vector<vector<T>> result(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            result[i].resize(data[i].size());
            for (size_t j = 0; j < data[i].size(); ++j) {
                result[i][j] = data[i][j] * other.data[i][j];
            }
        }
        return Tensor(result);
    }

    Tensor(const vector<int>& size) {
        dimensions = size;
        int totalSize = 1;
        for (int dim : size) {
            totalSize *= dim;
        }
        data = vector<T>(totalSize, 0);
    }

    vector<T> gradient() {
        return this -> grad;
    }

    
    vector<vector<T>> data;
    vector<vector<T>> grad;
    vector<int> dimensions;

};



#endif //TENSOR_H
