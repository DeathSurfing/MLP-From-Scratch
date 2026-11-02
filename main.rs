// A small multilayer perceptron written with the Rust standard library only.
//
// The network is 784 -> 128 -> 64 -> 10. It trains on the MNIST CSV format
// (label followed by 784 pixel values) and evaluates a saved weight file.
// There is no convolution and no external ML crate.

use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::process;

type Matrix = Vec<Vec<f32>>;

struct SimpleRng {
    seed: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.seed >> 40) as f32) / ((1u64 << 24) as f32)
    }

    fn range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }
}

struct Network {
    sizes: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Vec<f32>>,
}

impl Network {
    fn new(sizes: &[usize], rng: &mut SimpleRng) -> Self {
        let mut weights = Vec::with_capacity(sizes.len() - 1);
        let mut biases = Vec::with_capacity(sizes.len() - 1);

        for layer in 0..sizes.len() - 1 {
            let fan_in = sizes[layer];
            let fan_out = sizes[layer + 1];
            let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
            let mut rows = Vec::with_capacity(fan_out);
            for _ in 0..fan_out {
                let mut row = Vec::with_capacity(fan_in);
                for _ in 0..fan_in {
                    row.push(rng.range(-limit, limit));
                }
                rows.push(row);
            }
            weights.push(rows);
            biases.push(vec![0.0; fan_out]);
        }

        Self {
            sizes: sizes.to_vec(),
            weights,
            biases,
        }
    }

    fn forward(&self, input: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut activations = Vec::with_capacity(self.weights.len() + 1);
        let mut pre_activations = Vec::with_capacity(self.weights.len());
        activations.push(input.to_vec());

        for layer in 0..self.weights.len() {
            let mut z = vec![0.0f32; self.sizes[layer + 1]];
            for neuron in 0..self.sizes[layer + 1] {
                let mut sum = self.biases[layer][neuron];
                for (weight, activation) in self.weights[layer][neuron]
                    .iter()
                    .zip(activations[layer].iter())
                {
                    sum += weight * activation;
                }
                z[neuron] = sum;
            }

            let activation = if layer == self.weights.len() - 1 {
                softmax(&z)
            } else {
                z.iter().map(|&value| value.max(0.0)).collect()
            };
            pre_activations.push(z);
            activations.push(activation);
        }

        (activations, pre_activations)
    }

    fn predict(&self, input: &[f32]) -> usize {
        let (activations, _) = self.forward(input);
        let output = activations.last().expect("network has an output layer");
        output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap()
    }

    fn accuracy(&self, data: &[Vec<f32>], labels: &[u8]) -> f32 {
        let mut correct = 0usize;
        for (input, &label) in data.iter().zip(labels.iter()) {
            if self.predict(input) == label as usize {
                correct += 1;
            }
        }
        correct as f32 / data.len() as f32
    }

    fn train(
        &mut self,
        data: &[Vec<f32>],
        labels: &[u8],
        epochs: usize,
        learning_rate: f32,
        batch_size: usize,
        rng: &mut SimpleRng,
    ) {
        let mut indices: Vec<usize> = (0..data.len()).collect();
        let layer_count = self.weights.len();

        for epoch in 0..epochs {
            for i in (1..indices.len()).rev() {
                let j = (rng.next_f32() * (i + 1) as f32) as usize;
                indices.swap(i, j);
            }

            let mut epoch_loss = 0.0f64;
            let mut sample_count = 0usize;

            for batch in indices.chunks(batch_size) {
                let mut batch_loss = 0.0f64;
                let mut grad_weights: Vec<Matrix> = self
                    .weights
                    .iter()
                    .map(|layer| layer.iter().map(|row| vec![0.0; row.len()]).collect())
                    .collect();
                let mut grad_biases: Vec<Vec<f32>> = self
                    .biases
                    .iter()
                    .map(|layer| vec![0.0; layer.len()])
                    .collect();

                for &index in batch {
                    let input = &data[index];
                    let label = labels[index] as usize;
                    let (activations, pre_activations) = self.forward(input);

                    let output = &activations[layer_count];
                    batch_loss += -(output[label].max(1e-12) as f64).ln();

                    let mut delta = output.clone();
                    delta[label] -= 1.0;

                    for layer in (0..layer_count).rev() {
                        for neuron in 0..self.weights[layer].len() {
                            grad_biases[layer][neuron] += delta[neuron];
                            for (input_index, activation) in activations[layer].iter().enumerate() {
                                grad_weights[layer][neuron][input_index] +=
                                    delta[neuron] * activation;
                            }
                        }

                        if layer > 0 {
                            let mut previous_delta = vec![0.0f32; self.sizes[layer]];
                            for input_neuron in 0..self.sizes[layer] {
                                let mut sum = 0.0f32;
                                for neuron in 0..self.weights[layer].len() {
                                    sum +=
                                        self.weights[layer][neuron][input_neuron] * delta[neuron];
                                }
                                previous_delta[input_neuron] =
                                    if pre_activations[layer - 1][input_neuron] > 0.0 {
                                        sum
                                    } else {
                                        0.0
                                    };
                            }
                            delta = previous_delta;
                        }
                    }
                }

                let scale = learning_rate / batch.len() as f32;
                for layer in 0..layer_count {
                    for neuron in 0..self.weights[layer].len() {
                        for input_index in 0..self.weights[layer][neuron].len() {
                            self.weights[layer][neuron][input_index] -=
                                scale * grad_weights[layer][neuron][input_index];
                        }
                        self.biases[layer][neuron] -= scale * grad_biases[layer][neuron];
                    }
                }

                epoch_loss += batch_loss;
                sample_count += batch.len();
            }

            println!(
                "epoch {}/{}: mean loss {:.4}",
                epoch + 1,
                epochs,
                epoch_loss / sample_count as f64
            );
        }
    }

    fn save(&self, path: &str) -> io::Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        writeln!(writer, "{}", self.weights.len())?;

        for layer in 0..self.weights.len() {
            writeln!(writer, "{} {}", self.sizes[layer + 1], self.sizes[layer])?;
            for row in &self.weights[layer] {
                for (index, value) in row.iter().enumerate() {
                    if index > 0 {
                        write!(writer, " ")?;
                    }
                    write!(writer, "{:.17}", value)?;
                }
                writeln!(writer)?;
            }
        }

        for layer in &self.biases {
            writeln!(writer, "{}", layer.len())?;
            for (index, value) in layer.iter().enumerate() {
                if index > 0 {
                    write!(writer, " ")?;
                }
                write!(writer, "{:.17}", value)?;
            }
            writeln!(writer)?;
        }

        Ok(())
    }

    fn load(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut lines = BufReader::new(file).lines();

        let layer_count: usize = next_line(&mut lines)?
            .trim()
            .parse()
            .map_err(|_| invalid_data("layer count"))?;

        let mut weights = Vec::with_capacity(layer_count);
        let mut sizes = Vec::with_capacity(layer_count + 1);

        for layer in 0..layer_count {
            let dims: Vec<usize> = next_line(&mut lines)?
                .split_whitespace()
                .map(|value| value.parse().map_err(|_| invalid_data("layer dimensions")))
                .collect::<io::Result<_>>()?;
            if dims.len() != 2 {
                return Err(invalid_data("layer dimensions"));
            }
            if layer == 0 {
                sizes.push(dims[1]);
            }
            sizes.push(dims[0]);

            let mut rows = Vec::with_capacity(dims[0]);
            for _ in 0..dims[0] {
                let row: Vec<f32> = next_line(&mut lines)?
                    .split_whitespace()
                    .map(|value| value.parse().map_err(|_| invalid_data("weight")))
                    .collect::<io::Result<_>>()?;
                if row.len() != dims[1] {
                    return Err(invalid_data("weight row length"));
                }
                rows.push(row);
            }
            weights.push(rows);
        }

        let mut biases = Vec::with_capacity(layer_count);
        for layer in 0..layer_count {
            let size: usize = next_line(&mut lines)?
                .trim()
                .parse()
                .map_err(|_| invalid_data("bias count"))?;
            if size != sizes[layer + 1] {
                return Err(invalid_data("bias count"));
            }
            let values: Vec<f32> = next_line(&mut lines)?
                .split_whitespace()
                .map(|value| value.parse().map_err(|_| invalid_data("bias")))
                .collect::<io::Result<_>>()?;
            if values.len() != size {
                return Err(invalid_data("bias row length"));
            }
            biases.push(values);
        }

        Ok(Self {
            sizes,
            weights,
            biases,
        })
    }
}

fn next_line(lines: &mut impl Iterator<Item = io::Result<String>>) -> io::Result<String> {
    lines
        .next()
        .transpose()?
        .ok_or_else(|| invalid_data("unexpected end of weights file"))
}

fn invalid_data(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f32 = values.iter().map(|&value| (value - max).exp()).sum();
    values
        .iter()
        .map(|&value| (value - max).exp() / sum)
        .collect()
}

fn load_mnist(path: &str) -> io::Result<(Vec<Vec<f32>>, Vec<u8>)> {
    let file = File::open(path)?;
    let mut data = Vec::new();
    let mut labels = Vec::new();

    for (line_number, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let mut fields = line.split(',');
        let first = fields.next().unwrap_or("").trim();
        let label: u8 = match first.parse() {
            Ok(value) => value,
            Err(_) if line_number == 0 => continue, // optional header row
            Err(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("line {}: label is not a digit", line_number + 1),
                ))
            }
        };

        let pixels: Vec<f32> = fields
            .map(|value| value.trim().parse::<f32>().map(|pixel| pixel / 255.0))
            .collect::<Result<_, _>>()
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("line {}: pixel is not a number", line_number + 1),
                )
            })?;

        if pixels.len() != 28 * 28 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "line {}: expected 784 pixels, found {}",
                    line_number + 1,
                    pixels.len()
                ),
            ));
        }

        data.push(pixels);
        labels.push(label);
    }

    Ok((data, labels))
}

fn usage(program: &str) {
    eprintln!(
        "usage:\n  {program} train <train.csv> <weights.dat> [epochs] [learning-rate] [batch-size]\n  {program} eval <test.csv> <weights.dat>\n  {program} eval <test.csv> <weights.dat>   # default when no arguments are given"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_is_a_distribution() {
        let values = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn committed_weights_load_and_run() {
        let network = Network::load("weights.dat").expect("committed weights load");
        let (activations, _) = network.forward(&vec![0.0; 784]);
        let output = activations.last().expect("output layer");
        assert_eq!(output.len(), 10);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let program = args.first().map(String::as_str).unwrap_or("mnist-mlp");

    if args.len() == 1 {
        let (data, labels) = load_mnist("mnist_test.csv")?;
        let network = Network::load("weights.dat")?;
        let accuracy = network.accuracy(&data, &labels);
        println!("test accuracy: {:.2}%", accuracy * 100.0);
        return Ok(());
    }

    match args.get(1).map(String::as_str) {
        Some("train") => {
            if args.len() < 4 {
                usage(program);
                process::exit(2);
            }
            let (data, labels) = load_mnist(&args[2])?;
            let epochs = args
                .get(4)
                .and_then(|value| value.parse().ok())
                .unwrap_or(10);
            let learning_rate = args
                .get(5)
                .and_then(|value| value.parse().ok())
                .unwrap_or(0.1);
            let batch_size = args
                .get(6)
                .and_then(|value| value.parse().ok())
                .unwrap_or(32);
            let mut rng = SimpleRng::new(42);
            let mut network = Network::new(&[784, 128, 64, 10], &mut rng);

            println!(
                "training on {} examples: {} epochs, lr {}, batch {}",
                data.len(),
                epochs,
                learning_rate,
                batch_size
            );
            network.train(&data, &labels, epochs, learning_rate, batch_size, &mut rng);
            network.save(&args[3])?;
            println!("saved weights to {}", args[3]);
            Ok(())
        }
        Some("eval") => {
            if args.len() < 4 {
                usage(program);
                process::exit(2);
            }
            let (data, labels) = load_mnist(&args[2])?;
            let network = Network::load(&args[3])?;
            let accuracy = network.accuracy(&data, &labels);
            println!("test accuracy: {:.2}%", accuracy * 100.0);
            Ok(())
        }
        Some("-h") | Some("--help") => {
            usage(program);
            Ok(())
        }
        _ => {
            usage(program);
            process::exit(2);
        }
    }
}
