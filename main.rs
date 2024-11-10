use std::fs::File;
use std::io::{BufReader, BufRead, Write};
use std::path::Path;

fn load_mnist_data(filename: &str) -> std::io::Result<(Vec<Vec<f64>>, Vec<f64>)> {
    let mut data = Vec::new();
    let mut labels = Vec::new();
    let file = File::open(filename)?;

    for (i, line) in BufReader::new(file).lines().enumerate() {
        if i == 0 { continue; } // Skip header

        let line = line?;
        let values: Vec<f64> = line
            .split(',')
            .map(|x| x.trim().parse::<f64>().expect("Error parsing a value"))
            .collect();

        if values.len() > 1 {
            labels.push(values[0]);
            // Normalize pixel values to [0, 1]
            data.push(values[1..].iter().map(|&x| x / 255.0).collect());
        }
    }
    Ok((data, labels))
}

// Split data into training and test sets
fn split_data(data: Vec<Vec<f64>>, labels: Vec<f64>, test_ratio: f64) 
    -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let test_size = (data.len() as f64 * test_ratio) as usize;
    let train_size = data.len() - test_size;
    
    let mut indices: Vec<usize> = (0..data.len()).collect();
    
    // Shuffle indices
    let mut rng = SimpleRng::new(Some(42));  // Using a fixed seed for reproducibility
    for i in (1..indices.len()).rev() {
        let j = (rng.next() * (i + 1) as f64) as usize;
        indices.swap(i, j);
    }

    let mut train_data = Vec::with_capacity(train_size);
    let mut train_labels = Vec::with_capacity(train_size);
    let mut test_data = Vec::with_capacity(test_size);
    let mut test_labels = Vec::with_capacity(test_size);

    for &idx in &indices[..train_size] {
        train_data.push(data[idx].clone());
        train_labels.push(labels[idx]);
    }
    for &idx in &indices[train_size..] {
        test_data.push(data[idx].clone());
        test_labels.push(labels[idx]);
    }

    (train_data, train_labels, test_data, test_labels)
}

// Simple random number generator
struct SimpleRng {
    seed: u64,
}

impl SimpleRng {
    fn new(seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64
        });
        SimpleRng { seed }
    }

    fn next(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
        (self.seed & 0x7fffffff) as f64 / 0x7fffffff as f64
    }

    fn range(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.next()
    }
}

fn load_weights(filename: &str) -> std::io::Result<(Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>)> {
    let file = File::open(filename)?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    
    // Read number of layers
    reader.read_line(&mut line)?;
    let num_layers: usize = line.trim().parse().unwrap();
    line.clear();
    
    // Read weights
    let mut weights = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        // Read layer dimensions
        reader.read_line(&mut line)?;
        let dims: Vec<usize> = line
            .trim()
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        line.clear();
        
        let mut layer = Vec::with_capacity(dims[0]);
        
        // Read weight values
        for _ in 0..dims[0] {
            reader.read_line(&mut line)?;
            let neuron: Vec<f64> = line
                .trim()
                .split_whitespace()
                .map(|x| x.parse().unwrap())
                .collect();
            layer.push(neuron);
            line.clear();
        }
        
        weights.push(layer);
    }
    
    // Read biases
    let mut biases = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        reader.read_line(&mut line)?;
        let _size: usize = line.trim().parse().unwrap();
        line.clear();
        
        reader.read_line(&mut line)?;
        let layer_biases: Vec<f64> = line
            .trim()
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        biases.push(layer_biases);
        line.clear();
    }
    
    Ok((weights, biases))
}

// Activation function (ReLU)
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

// Softmax function
fn softmax(x: &[f64]) -> Vec<f64> {
    let max_val = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_sum: f64 = x.iter()
        .map(|&xi| (xi - max_val).exp())
        .sum();
    x.iter()
        .map(|&xi| (xi - max_val).exp() / exp_sum)
        .collect()
}

// Forward pass through the network
fn forward_pass(input: &[f64], weights: &[Vec<Vec<f64>>], biases: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut activations = vec![input.to_vec()];
    
    for layer in 0..weights.len() {
        let mut layer_output = vec![0.0; weights[layer].len()];
        
        // Compute weighted sum and add bias for each neuron
        for (neuron_idx, neuron_weights) in weights[layer].iter().enumerate() {
            let weighted_sum: f64 = neuron_weights.iter()
                .zip(activations[layer].iter())
                .map(|(&w, &a)| w * a)
                .sum();
            
            layer_output[neuron_idx] = if layer == weights.len() - 1 {
                weighted_sum + biases[layer][neuron_idx] // No ReLU on last layer
            } else {
                relu(weighted_sum + biases[layer][neuron_idx])
            };
        }
        
        // Apply softmax to final layer
        if layer == weights.len() - 1 {
            layer_output = softmax(&layer_output);
        }
        
        activations.push(layer_output);
    }
    
    activations
}

// Predict class (returns index of highest probability)
fn predict(output: &[f64]) -> usize {
    output.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}

// Calculate accuracy
fn calculate_accuracy(
    data: &[Vec<f64>],
    labels: &[f64],
    weights: &[Vec<Vec<f64>>],
    biases: &[Vec<f64>]
) -> f64 {
    let mut correct = 0;
    let total = data.len();
    
    for (input, &true_label) in data.iter().zip(labels.iter()) {
        let activations = forward_pass(input, weights, biases);
        let prediction = predict(&activations.last().unwrap());
        
        if prediction as f64 == true_label {
            correct += 1;
        }
    }
    
    correct as f64 / total as f64
}

// Modified main function to evaluate on test set
fn main() -> std::io::Result<()> {
    println!("Loading MNIST training dataset...");
    let (data, labels) = load_mnist_data("mnist_train.csv")?;
    println!("Training dataset loaded successfully!");
    
    println!("Loading MNIST test dataset...");
    let (test_data, test_labels) = load_mnist_data("mnist_test.csv")?;
    println!("Test dataset loaded successfully!");
    
    let layer_sizes = vec![784, 128, 64, 10];
    
    // Load trained weights
    println!("Loading trained weights...");
    let (weights, biases) = match load_weights("weights.dat") {
        Ok((w, b)) => (w, b),
        Err(e) => {
            println!("Error loading weights: {}. Cannot evaluate model.", e);
            return Ok(());
        }
    };
    
    // Calculate and print accuracy on test set
    println!("Calculating accuracy on test set...");
    let test_accuracy = calculate_accuracy(&test_data, &test_labels, &weights, &biases);
    println!("Test accuracy: {:.2}%", test_accuracy * 100.0);
    
    Ok(())
}

