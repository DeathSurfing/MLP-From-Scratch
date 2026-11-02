# MNIST Multilayer Perceptron from Scratch in Rust

A 784-128-64-10 multilayer perceptron written with the Rust standard library
only. The forward pass, backward pass, softmax cross-entropy loss, and
mini-batch SGD are implemented by hand. There is no PyTorch, no ndarray, and no
other ML crate.

This repository started as a CNN learning experiment. The code that survived is
an MLP, not a convolutional network: there are no convolution or pooling layers.
The project is now named and documented for what it actually is.

## Result

Trained on the 60,000-image MNIST training set for 12 epochs with mini-batch
SGD (batch size 32, learning rate 0.1, Xavier-uniform initialization, seed 42):

```text
test accuracy: 97.46%
```

The committed `weights.dat` is the checkpoint that produced that number. The
evaluation is deterministic.

## Architecture

```text
784 inputs -> 128 hidden (ReLU) -> 64 hidden (ReLU) -> 10 outputs (softmax)
```

- Loss: softmax cross-entropy.
- Optimizer: mini-batch stochastic gradient descent.
- Initialization: Xavier uniform, zero biases.
- Weight layout: one row per neuron, one value per input connection.

## Reproduce

### 1. Install Rust

<https://www.rust-lang.org/tools/install>

### 2. Get the MNIST CSV files

The loader accepts the common CSV format with one label and 784 pixel values per
row. A header row is optional. A public mirror of the MNIST CSVs is:

```bash
curl -LO https://data.pjreddie.com/files/mnist_train.csv
curl -LO https://data.pjreddie.com/files/mnist_test.csv
```

### 3. Train

```bash
cargo run --release -- train mnist_train.csv weights.dat 12 0.1 32
```

Arguments are `train <csv> <weights-out> [epochs] [learning-rate] [batch-size]`.
The command prints the mean loss for each epoch and writes the weight file.

### 4. Evaluate

```bash
cargo run --release -- eval mnist_test.csv weights.dat
```

Expected output:

```text
test accuracy: 97.46%
```

Running the binary with no arguments evaluates `mnist_test.csv` against
`weights.dat`. The old binary also required the training CSV for evaluation;
the new default does not.

## Repository layout

```text
Cargo.toml    package manifest
Cargo.lock    locked dependency graph
main.rs       data loader, network, training loop, evaluation, weight I/O
weights.dat   trained checkpoint (97.46% test accuracy)
README.md     this file
LICENSE       MIT
```

## What this project demonstrates

- A hand-written forward pass and backpropagation for a dense network.
- A from-scratch training loop with mini-batch SGD.
- A documented, reproducible training and evaluation workflow.
- A measured result, not an asserted one.

It does not demonstrate convolutional layers, pooling, or a CNN. The old
"CNN from scratch" description was inaccurate and has been removed.

## License

MIT. See [LICENSE](LICENSE).
