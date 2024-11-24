#include <iostream>
#include "Data/DataLoader.h"
#include "Data/DataSet.h"

int main() {
    DataSet minstdata;
    minstdata.readMnistData();
    return 0;
}
