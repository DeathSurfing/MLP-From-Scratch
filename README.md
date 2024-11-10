# MNIST Handwritten Digit Recognition in Rust

This project implements a neural network that performs handwritten digit classification using the MNIST dataset. The code is written in Rust and loads the dataset, preprocesses the data, and evaluates the model on the test set.

## Features
- Load MNIST data from CSV files.
- Normalize pixel values to the range [0, 1].
- Train a neural network on the training data.
- Evaluate the trained model on the test set.
- Calculate accuracy of predictions on the test set.

## Prerequisites
- Rust programming language (version 1.56 or higher).
- MNIST dataset in CSV format (downloadable from Kaggle).
- Pretrained weights (weights.dat file).

## Dataset
You need to download the MNIST dataset in CSV format from Kaggle. The dataset can be found [here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).

- **mnist_train.csv**: This file contains the training data with pixel values (each row represents an image of a digit).
- **mnist_test.csv**: This file contains the test data with pixel values.

Once downloaded, place the following files in the project directory:
- `mnist_train.csv`
- `mnist_test.csv`

## Weights File
The neural network requires a file `weights.dat` containing the pretrained weights and biases for the network. The weights should be loaded from this file in the code and used during evaluation. This file is assumed to be already generated from a previous training process.

## Setup
To build and run this project, follow the steps below:

### 1. Install Rust
If you haven't installed Rust yet, you can do so by following the instructions at: https://www.rust-lang.org/tools/install

### 2. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```
### 3. Build the Project
Run the following command to build the project:

```bash
cargo build --release
```
### 4. Run the Project
Once the project is built, you can run it using the following command:

```bash
cargo run
```
### 5. View the Results
The program will load the training and test datasets, load the pretrained weights, and calculate the accuracy of the model on the test dataset. The accuracy will be displayed in the terminal.

File Structure
```bash
/src
  main.rs           - Main code implementing the neural network and data processing.
  /weights.dat      - Pretrained weights and biases.
  /mnist_train.csv  - MNIST training dataset.
  /mnist_test.csv   - MNIST test dataset.
```
License
This project is licensed under the MIT License - see the LICENSE file for details.
