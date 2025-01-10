
//
// Created by siyan on 2024/10/27.
//

#include "DataSet.h"

DataSet::DataSet()
{
}

void DataSet::readMnistTrainLable()
{
    label = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    std::ifstream ifsLable;
    ifsLable.open("../datasets/FashionMNIST/train-labels.idx1-ubyte", std::ios::in | std::ios::binary);

    unsigned char bytes[8];
    ifsLable.read((char *)bytes, 8);
    uint32_t magic = (uint32_t)((bytes[0] << 24) |
                                (bytes[1] << 16) |
                                (bytes[2] << 8) |
                                bytes[3]);
    uint32_t num = (uint32_t)((bytes[4] << 24) |
                              (bytes[5] << 16) |
                              (bytes[6] << 8) |
                              bytes[7]);
    train_output.resize(num,10);
    train_output.setZero();
    int cnt = 0;
    while (!ifsLable.eof())
    {
        unsigned char byte;
        ifsLable.read((char *)&byte, 1);
        if (ifsLable.fail())
        {
            break;
        }
        int pos = (unsigned int)byte;
        train_output(cnt,pos) = 1;
        cnt++;
    }
    ifsLable.close();
}

void DataSet::readMnistTestLable()
{
    label = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    std::ifstream ifsLable;
    ifsLable.open("../datasets/FashionMNIST/t10k-labels.idx1-ubyte", std::ios::in | std::ios::binary);
    unsigned char bytes[8];
    ifsLable.read((char *)bytes, 8);
    uint32_t magic = (uint32_t)((bytes[0] << 24) |
                                (bytes[1] << 16) |
                                (bytes[2] << 8) |
                                bytes[3]);
    uint32_t num = (uint32_t)((bytes[4] << 24) |
                              (bytes[5] << 16) |
                              (bytes[6] << 8) |
                              bytes[7]);

    test_output.resize(num,10);
    test_output.setZero();
    int cnt = 0;
    while (!ifsLable.eof())
    {
        unsigned char byte;
        ifsLable.read((char *)&byte, 1);
        if (ifsLable.fail())
        {
            break;
        }
        int pos = (unsigned int)byte;
        test_output(cnt,pos) = 1;
        cnt++;
    }
    ifsLable.close();
}

void DataSet::readMnistTrainImage()
{
    std::ifstream ifsLable;
    ifsLable.open("../datasets/FashionMNIST/train-images.idx3-ubyte", std::ios::in | std::ios::binary);
    unsigned char bytes[16];
    ifsLable.read((char *)bytes, 16);
    uint32_t magic = (uint32_t)((bytes[0] << 24) |
                                (bytes[1] << 16) |
                                (bytes[2] << 8) |
                                bytes[3]);
    uint32_t num = (uint32_t)((bytes[4] << 24) |
                              (bytes[5] << 16) |
                              (bytes[6] << 8) |
                              bytes[7]);
    uint32_t rows = (uint32_t)((bytes[8] << 24) |
                               (bytes[9] << 16) |
                               (bytes[10] << 8) |
                               bytes[11]);
    uint32_t cols = (uint32_t)((bytes[12] << 24) |
                               (bytes[13] << 16) |
                               (bytes[14] << 8) |
                               bytes[15]);
    //printf("MnistTrainImage %d %d %d %d\n", magic, num, rows, cols);
    train_input.resize(num, (rows * cols));
    train_input.setZero();
    int Num = 0;
    while (!ifsLable.eof() && Num<num)
    {
        int cnt = 0;
        while (cnt < rows * cols && !ifsLable.fail())
        {
            unsigned char byte;
            ifsLable.read((char *)&byte, 1);
            int pix = (unsigned int)byte;
            train_input(Num,cnt) = pix;
            ++cnt;
        }
        ++Num;
    }
    ifsLable.close();
}

void DataSet::readMnistTestImage()
{
    std::ifstream ifsLable;
    ifsLable.open("../datasets/FashionMNIST/t10k-images.idx3-ubyte", std::ios::in | std::ios::binary);
    unsigned char bytes[16];
    ifsLable.read((char *)bytes, 16);
    uint32_t magic = (uint32_t)((bytes[0] << 24) |
                                (bytes[1] << 16) |
                                (bytes[2] << 8) |
                                bytes[3]);
    uint32_t num = (uint32_t)((bytes[4] << 24) |
                              (bytes[5] << 16) |
                              (bytes[6] << 8) |
                              bytes[7]);
    uint32_t rows = (uint32_t)((bytes[8] << 24) |
                               (bytes[9] << 16) |
                               (bytes[10] << 8) |
                               bytes[11]);
    uint32_t cols = (uint32_t)((bytes[12] << 24) |
                               (bytes[13] << 16) |
                               (bytes[14] << 8) |
                               bytes[15]);
    // printf("MnistTestImage %d %d %d %d\n", magic, num, rows, cols);
    test_input.resize(num, (rows * cols));
    test_input.setZero();
    int Num = 0;
    while (!ifsLable.eof() && Num<num)
    {
        int cnt = 0;
        while (cnt < rows * cols && !ifsLable.fail())
        {
            unsigned char byte;
            ifsLable.read((char *)&byte, 1);
            int pix = (unsigned int)byte;
            test_input(Num,cnt) = pix;
            ++cnt;
        }
        Num++;
    }
    ifsLable.close();
}

void DataSet::printDigit(Eigen::VectorXd x, double mask)
{
    if (x.size() != 28 * 28)
    {
        printf("printDigit Error\n");
        return;
    }
    for (int i = 0; i < 28; ++i)
    {
        for (int j = 0; j < 28; ++j)
        {
            if (x[i * 28 + j] > mask)
            {
                printf("##");
            }
            else
            {
                printf("  ");
            }
        }
        printf("\n");
    }
}

void DataSet::readMnistData()
{
    readMnistTrainLable();
    readMnistTrainImage();
    readMnistTestImage();
    readMnistTestLable();
    printf("train_image = %ld train_lable = %ld \n", train_input.rows(), train_output.rows());
    printf("test_image = %ld test_lable = %ld \n", test_input.rows(), test_output.rows());
}

DataSet::~DataSet()
{
}

Eigen::MatrixXd DataSet::getTrainInput()
{
    return train_input;
}

Eigen::MatrixXd DataSet::getTrainOutput()
{
    return train_output;
}

Eigen::MatrixXd DataSet::getTestInput()
{
    return test_input;
}

Eigen::MatrixXd DataSet::getTestOutput()
{
    return test_output;
}

// Eigen::MatrixXd DataSet::getNormalizedData(Eigen::MatrixXd data)
// {
//     Eigen::MatrixXd data_normalized = data.rowwise() / data.rowwise().norm();
//     return data_normalized;
// }
