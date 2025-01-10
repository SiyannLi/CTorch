
//
// Created by siyan on 2024/10/27.
//

#ifndef DATALOADER_H
#define DATALOADER_H

#include <cstdlib>  // For rand() and srand()
#include <ctime>    // For time()
#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <Eigen>

template <typename T>
void printData(T t);

template <typename T>
int maxIndex(T t);

template <typename T>
void saveLogs(std::string path, std::vector<T> logs);

void shuffleData(Eigen::MatrixXd &train_input, Eigen::MatrixXd &train_output);

#endif //DATALOADER_H
