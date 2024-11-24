# CTorch Project

## Project Overview
CTorch is a C++-based tool designed for training and testing neural networks. The project leverages the Eigen library for efficient matrix operations and mathematical computations. It aims to provide a high-performance environment for neural network development.

## Dependencies

1. **C++ Standard**: Requires support for C++20 or later.
2. **Eigen Library**: The Eigen library must be downloaded and placed in the root directory of the project workspace.

## Project Structure
Ensure the root directory of the project has the following structure:
```
CTorch/
├── CMakeLists.txt
├── build/
├── Eigen/  # Eigen library should be manually downloaded and placed here
└── ...
```

## Build and Run

### Build Steps
Run the following commands in a terminal to build and run the project:
```bash
mkdir build
cd build
cmake ..
make
```

### Running the Project
After successful compilation, execute the following command to run the program:
```bash
./CTorch
```
You can see the forward output of the dummy network at the terminal.

## Notes
- Ensure that the C++ compiler supports the C++20 standard.
- Verify that the Eigen library is correctly placed in the project root directory.

## Feedback and Support
For any issues, please submit an issue or contact us via email.


