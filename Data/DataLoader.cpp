//
// Created by 刘禹恒 on 2025/1/10.
//
# include "DataLoader.h"

template <typename T>
void printData(T t)
{
    for (int i = 0; i < t.size(); ++i)
    {
        printf("%f ", t[i]);
    }
    printf("\n");
}

template <typename T>
int maxIndex(T t)
{
    return std::max_element(t.begin(), t.end()) - t.begin();
}

template <typename T>
void saveLogs(std::string path, std::vector<T> logs)
{
    std::ofstream ofs(path, std::ios::out);
    if (!ofs.is_open())
    {
        printf("save logs Error\n");
    }
    else
    {
        for (auto a : logs)
        {
            ofs << a << std::endl;
        }
        ofs.close();
    }
}

void shuffleData(Eigen::MatrixXd& train_input, Eigen::MatrixXd& train_output)
{
    int len = train_input.rows();
    int num = len / 4;
    srand(static_cast<unsigned int>(time(0)));
    while (num--)
    {
        int p1 = rand() % len;
        int p2 = rand() % len;
        if (p1 >= len || p2 >= len)
        {
            continue;
        }
        Eigen::VectorXd temp = train_input.row(p1);
        train_input.row(p1) = train_input.row(p2);
        train_input.row(p2) = temp;
        temp = train_output.row(p1);
        train_output.row(p1) = train_output.row(p2);
        train_output.row(p2) = temp;
    }
}