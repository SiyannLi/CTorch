#include <iostream>
#include "Data/DataLoader.h"
#include "Data/DataSet.h"

int main() {
    DataSet minstdata;
    minstdata.readMnistData();
    std::vector<std::vector<double>> test_input = minstdata.getTestInput();
    minstdata.printDigit(test_input[0],0);
    return 0;
}
