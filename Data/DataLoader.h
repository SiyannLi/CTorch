
//
// Created by siyan on 2024/10/27.
//

#ifndef DATALOADER_H
#define DATALOADER_H

#include <iostream>
#include <algorithm>
#include <fstream>

template <typename T>
void printData(T t);

template <typename T>
int maxIndex(T t);

template <typename T>
void saveLogs(std::string path, std::vector<T> logs);

template <typename T>
void shuffleData(T &train_input, T &train_output);

#endif //DATALOADER_H
